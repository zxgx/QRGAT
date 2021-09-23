import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from modules import EncoderRNN, GNN, DecoderRNN


class Graph2SeqOutput:
    def __init__(self, encoder_outputs, encoder_state, decode_tokens):
        self.encoder_outputs = encoder_outputs
        self.encoder_state = encoder_state
        self.decode_tokens = decode_tokens
        self.enc_attn_weights = None
        self.pointer_probs = None
        self.nll_loss = 0.


class QGModel(nn.Module):
    def __init__(self, vocab, levi_graph, device, max_dec_step, word_dim, word_dropout, encoder_hidden_dim,
                 bidir_encoder, encoder_layer, encoder_dropout, answer_indicator_dim, gnn_hops, direction,
                 decoder_hidden_dim):
        super(QGModel, self).__init__()
        self.vocab = vocab
        self.levi_graph = levi_graph
        self.device = device
        self.max_dec_step = max_dec_step

        self.word_embedding = nn.Embedding(len(vocab), word_dim, padding_idx=0)
        self.word_dropout = nn.Dropout(word_dropout)  # Modified
        self.node_name_encoder = EncoderRNN(word_dim, encoder_hidden_dim, bidir_encoder, encoder_layer)
        self.edge_name_encoder = EncoderRNN(word_dim, encoder_hidden_dim, bidir_encoder, encoder_layer)
        self.encoder_dropout = nn.Dropout(encoder_dropout)
        self.answer_indicator_embedding = nn.Embedding(3, answer_indicator_dim, padding_idx=0)

        graph_dim = encoder_hidden_dim + answer_indicator_dim
        self.graph_encoder = GNN(gnn_hops, direction, levi_graph, graph_dim)

        self.encoder_decoder_adapter = nn.ModuleList([nn.Linear(graph_dim, decoder_hidden_dim) for _ in range(2)])

        self.decoder = DecoderRNN(device, word_dim, graph_dim, decoder_hidden_dim, encoder_dropout, self.word_embedding)

    def forward(self, batch, target_tensor=None, forcing_ratio=0, saved_out=None, visualize=False, sample=False,
                compute_loss=False, loss_reduction=True, eps_label_smooth=0.):
        graphs = batch.graphs
        batch_size, node_size = graphs['node_name_words'].shape[:2]
        edge_size = graphs['edge_name_words'].shape[1]
        num_nodes, num_edges = graphs['num_nodes'], graphs['num_edges']
        ext_vocab_size = batch.oov_dict.ext_vocab_size

        if self.levi_graph:
            max_node_size = graphs['max_node_size']
            num_virtual_nodes = num_nodes + num_edges
            input_mask = self.create_mask(num_virtual_nodes, max_node_size)  # batch size, max node size
            input_node_mask = self.create_mask(num_nodes, max_node_size)
        else:
            max_node_size = node_size
            input_mask = self.create_mask(num_nodes, max_node_size)  # batch size, max node size
            input_node_mask = self.create_mask(num_nodes, max_node_size)

        if target_tensor is None:
            target_length = self.max_dec_step
        else:
            target_tensor = target_tensor.transpose(1, 0)  # max seq len, batch size
            target_length = target_tensor.shape[0]

        if saved_out:
            encoder_outputs = saved_out.encoder_outputs  # node + edge size, batch size, graph dim
            encoder_state = saved_out.encoder_state  # [1, batch size, graph dim]
        else:
            # batch size, node size, max node len, word dim
            node_name_word_emb = self.word_embedding(self.filter_oov(graphs['node_name_words'], ext_vocab_size))
            node_name_word_emb = self.word_dropout(node_name_word_emb).view(
                -1, node_name_word_emb.size(-2), node_name_word_emb.size(-1)
            )
            _, (node_name_emb, _) = self.node_name_encoder(node_name_word_emb, graphs['node_name_lens'].view(-1))
            # batch size, node size, encoder dim
            node_name_emb = self.encoder_dropout(node_name_emb.view(batch_size, node_size, -1))

            # batch size, node size, ans dim
            node_indicator = self.answer_indicator_embedding(graphs['answer_indicator'])
            # batch size, node size, encoder + ans dim = graph dim
            node_name_emb = torch.cat([node_name_emb, node_indicator], dim=-1)

            edge_name_word_emb = self.word_embedding(self.filter_oov(graphs['edge_name_words'], ext_vocab_size))
            edge_name_word_emb = self.word_dropout(edge_name_word_emb).view(
                -1, edge_name_word_emb.size(-2), edge_name_word_emb.size(-1)
            )
            _, (edge_name_emb, _) = self.edge_name_encoder(edge_name_word_emb, graphs['edge_name_lens'].view(-1))
            # batch size, edge size, encoder dim
            edge_name_emb = self.encoder_dropout(edge_name_emb.view(batch_size, edge_size, -1))

            # batch size, edge size, ans dim
            edge_indicator = torch.zeros(batch_size, edge_size, node_indicator.shape[-1]).to(self.device)
            # batch size, edge size, encoder + ans dim = graph dim
            edge_name_emb = torch.cat([edge_name_emb, edge_indicator], dim=-1)

            if self.levi_graph:
                # batch size, node + edge size, graph dim
                init_node_vec = self.gather(node_name_emb, edge_name_emb, num_nodes, num_edges, max_node_size)
                init_edge_vec = None
            else:
                init_node_vec = node_name_emb  # batch size, node size, graph dim
                init_edge_vec = edge_name_emb  # batch size, edge size, graph dim

            node_emb, graph_emb = self.graph_encoder(
                init_node_vec, init_edge_vec, graphs['node2edge'], graphs['edge2node'], node_mask=input_mask
            )
            encoder_outputs = node_emb  # node + edge size, batch size, graph dim
            encoder_state = (graph_emb, graph_emb)  # [1, batch size, graph dim]

        output = Graph2SeqOutput(encoder_outputs, encoder_state, torch.zeros(target_length, batch_size).long())
        if visualize:
            output.enc_attn_weights = torch.zeros(target_length, batch_size, max_node_size)
            output.pointer_probs = torch.zeros(target_length, batch_size)

        # [ 1, batch size, decoder dim ]
        decoder_state = tuple([self.encoder_decoder_adapter[i](x) for i, x in enumerate(encoder_state)])

        enc_context = None
        decoder_input = torch.tensor([self.vocab.SOS] * batch_size).to(self.device)
        for di in range(target_length):
            decoder_emb = self.word_embedding(self.filter_oov(decoder_input, ext_vocab_size))  # batch size, word dim
            decoder_emb = self.word_dropout(decoder_emb)
            decoder_output, decoder_state, dec_enc_attn, dec_prob_ptr, enc_context = self.decoder(
                decoder_emb, decoder_state, encoder_outputs, input_mask, input_node_mask, graphs['oov_idx'],
                ext_vocab_size, enc_context
            )

            if not sample:
                _, top_idx = decoder_output.topk(1)  # batch size, 1
            else:
                top_idx = torch.multinomial(decoder_output, 1)
            top_idx = top_idx.squeeze(1).detach()  # batch size
            output.decode_tokens[di] = top_idx

            if random.random() < forcing_ratio:
                decoder_input = target_tensor[di]
            else:
                decoder_input = top_idx

            if visualize:
                output.enc_attn_weights[di] = dec_enc_attn.detach().cpu()
                output.pointer_probs[di] = dec_prob_ptr.squeeze(1).detach().cpu()

            if compute_loss:
                if target_tensor is None:
                    gold = top_idx
                else:
                    gold = target_tensor[di]  # batch size
                decoder_output = torch.log(decoder_output + 1e-20)

                n_class = decoder_output.shape[1]
                # batch size, ext vocab size
                weight_dist = torch.zeros_like(decoder_output).scatter(1, gold.view(-1, 1), 1.)
                weight_dist = weight_dist * (1-eps_label_smooth) + (1-weight_dist) * eps_label_smooth / (n_class-1)
                non_pad_mask = gold.ne(self.vocab.PAD).float()
                nll_loss = - (weight_dist * decoder_output).sum(dim=1) * non_pad_mask  # batch size

                if loss_reduction:
                    nll_loss = nll_loss.sum() / non_pad_mask.sum()
                output.nll_loss += nll_loss

        return output

    def create_mask(self, tensor, dim):
        mask = torch.zeros(tensor.shape[0], dim)
        for i in range(tensor.shape[0]):
            mask[i, :tensor[i]] = 1
        return mask.to(self.device)

    def filter_oov(self, x, ext_vocab_size):
        if ext_vocab_size > len(self.vocab):
            x = x.clone()
            x[x >= len(self.vocab)] = self.vocab.UNK  # no clone
        return x

    def gather(self, node_name_emb, edge_name_emb, num_nodes, num_edges, max_node_size):
        # compact node tensors

        # batch size, node + edge size, graph dim
        tensor = torch.cat([node_name_emb, edge_name_emb], dim=1)
        node_size = node_name_emb.shape[1]

        index_tensor = []
        for i in range(tensor.shape[0]):
            selected_index = list(range(num_nodes[i])) + list(range(node_size, node_size+num_edges[i].item()))
            if len(selected_index) < max_node_size:
                selected_index += [max_node_size-1 for _ in range(max_node_size - len(selected_index))]
            index_tensor.append(selected_index)
        # batch size, max node size
        index_tensor = torch.tensor(index_tensor).to(self.device)
        # batch size, node + edge size, graph dim
        index_tensor = index_tensor.unsqueeze(-1).expand(-1, -1, tensor.shape[-1])
        return torch.gather(tensor, 1, index_tensor)

    def beam_search(self, batch, max_out_len, min_out_len, beam_size):
        graphs = batch.graphs
        batch_size, node_size = graphs['node_name_words'].shape[:2]
        edge_size = graphs['edge_name_words'].shape[1]
        num_nodes, num_edges = graphs['num_nodes'], graphs['num_edges']
        ext_vocab_size = batch.oov_dict.ext_vocab_size

        if self.levi_graph:
            max_node_size = graphs['max_node_size']
            num_virtual_nodes = num_nodes + num_edges
            input_mask = self.create_mask(num_virtual_nodes, max_node_size)  # batch size, max node size
            input_node_mask = self.create_mask(num_nodes, max_node_size)
        else:
            max_node_size = node_size
            input_mask = self.create_mask(num_nodes, max_node_size)  # batch size, max node size
            input_node_mask = self.create_mask(num_nodes, max_node_size)

        if max_out_len is None:
            max_out_len = self.max_dec_step - 1

        # batch size, node size, max node len, word dim
        node_name_word_emb = self.word_embedding(self.filter_oov(graphs['node_name_words'], ext_vocab_size))
        node_name_word_emb = self.word_dropout(node_name_word_emb).view(
            -1, node_name_word_emb.size(-2), node_name_word_emb.size(-1)
        )
        _, (node_name_emb, _) = self.node_name_encoder(node_name_word_emb, graphs['node_name_lens'].view(-1))
        # batch size, node size, encoder dim
        node_name_emb = self.encoder_dropout(node_name_emb.view(batch_size, node_size, -1))

        # batch size, node size, ans dim
        node_indicator = self.answer_indicator_embedding(graphs['answer_indicator'])
        # batch size, node size, encoder + ans dim = graph dim
        node_name_emb = torch.cat([node_name_emb, node_indicator], dim=-1)

        edge_name_word_emb = self.word_embedding(self.filter_oov(graphs['edge_name_words'], ext_vocab_size))
        edge_name_word_emb = self.word_dropout(edge_name_word_emb).view(
            -1, edge_name_word_emb.size(-2), edge_name_word_emb.size(-1)
        )
        _, (edge_name_emb, _) = self.edge_name_encoder(edge_name_word_emb, graphs['edge_name_lens'].view(-1))
        # batch size, edge size, encoder dim
        edge_name_emb = self.encoder_dropout(edge_name_emb.view(batch_size, edge_size, -1))

        # batch size, edge size, ans dim
        edge_indicator = torch.zeros(batch_size, edge_size, node_indicator.shape[-1]).to(self.device)
        # batch size, edge size, encoder + ans dim = graph dim
        edge_name_emb = torch.cat([edge_name_emb, edge_indicator], dim=-1)

        if self.levi_graph:
            # batch size, node + edge size, graph dim
            init_node_vec = self.gather(node_name_emb, edge_name_emb, num_nodes, num_edges, max_node_size)
            init_edge_vec = None
        else:
            init_node_vec = node_name_emb  # batch size, node size, graph dim
            init_edge_vec = edge_name_emb  # batch size, edge size, graph dim

        node_emb, graph_emb = self.graph_encoder(
            init_node_vec, init_edge_vec, graphs['node2edge'], graphs['edge2node'], node_mask=input_mask
        )
        encoder_outputs = node_emb  # node + edge size, batch size, graph dim
        encoder_state = (graph_emb, graph_emb)  # [1, batch size, graph dim]

        # [ 1, batch size, decoder dim ]
        decoder_state = tuple([self.encoder_decoder_adapter[i](x) for i, x in enumerate(encoder_state)])

        # Beam search decoding
        batch_results = []
        for batch_idx in range(batch_size):
            # batch size -> beam size
            beam_encoder_outputs = encoder_outputs[:, batch_idx: batch_idx+1].expand(-1, beam_size, -1)
            beam_oov_idx = graphs['oov_idx'][batch_idx:batch_idx+1].expand(beam_size, -1)
            beam_input_mask = input_mask[batch_idx: batch_idx+1].expand(beam_size, -1)
            beam_input_node_mask = input_node_mask[batch_idx: batch_idx+1].expand(beam_size, -1)
            single_decoder_state = tuple([each[:, batch_idx: batch_idx+1] for each in decoder_state])

            hypos = [Hypothesis([self.vocab.SOS], [], 1, single_decoder_state)]
            step = 0
            results, backup_results = [], []
            enc_context = None
            while len(hypos) > 0 and step <= max_out_len:
                if len(hypos) < beam_size:
                    hypos.extend(hypos[-1] for _ in range(beam_size - len(hypos)))
                beam_decoder_input = torch.tensor([h.tokens[-1] for h in hypos]).to(self.device)
                beam_decoder_state = (
                    torch.cat([h.decoder_state[0] for h in hypos], dim=1),  # 1, beam size, graph dim
                    torch.cat([h.decoder_state[1] for h in hypos], dim=1)   # 1, beam size, graph dim
                )
                # beam size, word dim
                beam_decoder_emb = self.word_embedding(self.filter_oov(beam_decoder_input, ext_vocab_size))
                beam_decoder_output, beam_decoder_state, dec_enc_attn, dec_prob_ptr, enc_context = self.decoder(
                    beam_decoder_emb, beam_decoder_state, beam_encoder_outputs, beam_input_mask, beam_input_node_mask,
                    beam_oov_idx, ext_vocab_size, enc_context
                )

                top_v, top_i = beam_decoder_output.topk(beam_size)  # beam size, beam size
                new_hypos = []
                for in_idx in range(len(hypos)):
                    for out_idx in range(beam_size):
                        new_prob, new_tok = top_v[in_idx][out_idx].item(), top_i[in_idx][out_idx].item()
                        non_word = new_tok == self.vocab.EOS

                        tmp_decoder_state = [x[0][in_idx].unsqueeze(0).unsqueeze(0) for x in beam_decoder_state]
                        new_hypos.append(hypos[in_idx].create_next(new_tok, new_prob, non_word, tmp_decoder_state))

                new_hypos = sorted(new_hypos, key=lambda h: -h.avg_log_prob)[:beam_size]
                hypos, complete_res, incomplete_res = [], [], []
                for hypo in new_hypos:
                    length = len(hypo)
                    if hypo.tokens[-1] == self.vocab.EOS:
                        if len(complete_res) < beam_size and min_out_len <= length <= max_out_len:
                            complete_res.append(hypo)
                    elif len(hypos) < beam_size and length < max_out_len:
                        hypos.append(hypo)
                    elif length == max_out_len and len(incomplete_res) < beam_size:
                        incomplete_res.append(hypo)

                if complete_res:
                    results.extend(complete_res)
                elif incomplete_res:
                    backup_results.extend(incomplete_res)
                step += 1

            if not results:
                results += backup_results
            batch_results.append(sorted(results, key=lambda h: -h.avg_log_prob)[: beam_size])
        return batch_results  # batch size, beam size, max out len


class Hypothesis(object):
    def __init__(self, tokens, log_probs, num_non_words, decoder_state):
        self.tokens = tokens  # type: List[int]
        self.log_probs = log_probs  # type: List[float]
        self.num_non_words = num_non_words  # type: int
        self.decoder_state = decoder_state

    def __repr__(self):
        return repr(self.tokens)

    def __len__(self):
        return len(self.tokens) - self.num_non_words

    @property
    def avg_log_prob(self):
        return sum(self.log_probs) / len(self.log_probs)

    def create_next(self, token, log_prob, non_word, decoder_state):
        return Hypothesis(
            tokens=self.tokens + [token], log_probs=self.log_probs + [log_prob],
            num_non_words=self.num_non_words + 1 if non_word else self.num_non_words,
            decoder_state=decoder_state
        )
