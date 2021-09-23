import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderRNN(nn.Module):
    def __init__(self, device, word_dim, graph_dim, decoder_hidden_dim, dropout, word_embedding):
        super(DecoderRNN, self).__init__()
        self.device = device
        self.fc_decoder_input = nn.Linear(word_dim + graph_dim, word_dim)
        self.lstm = nn.LSTM(word_dim, decoder_hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.attn_encoder_linear = nn.Linear(graph_dim, decoder_hidden_dim, bias=False)
        self.attn_decoder_linear = nn.Linear(decoder_hidden_dim * 2, decoder_hidden_dim, bias=False)
        self.attn_linear = nn.Linear(decoder_hidden_dim, 1, bias=False)

        combined_size = decoder_hidden_dim + graph_dim
        self.pre_out = nn.Linear(combined_size, word_dim, bias=False)
        # self.out = nn.Linear(word_dim, vocab_size, bias=False)

        self.pointer = nn.Linear(combined_size + word_dim + decoder_hidden_dim, 1)
        self.word_embedding = word_embedding

    def forward(self, x, decoder_state, encoder_outputs, input_mask, input_node_mask, encoder_word_idx, ext_vocab_size,
                prev_encoder_context):
        batch_size = x.shape[0]

        if prev_encoder_context is None:  # batch size ,graph dim
            prev_encoder_context = torch.zeros(batch_size, encoder_outputs.shape[-1], dtype=torch.float).to(self.device)
        decoder_x = self.fc_decoder_input(torch.cat([x, prev_encoder_context], dim=-1))  # batch size, word dim

        decoder_output, decoder_state = self.lstm(decoder_x.unsqueeze(0), decoder_state)
        decoder_output = self.dropout(decoder_output)  # 1, batch size, decoder dim

        decoder_hidden = self.dropout(torch.cat(decoder_state, dim=-1)).squeeze(0)  # batch size, decoder dim * 2

        # batch size, max node size
        enc_energy = self.enc_attn_fn(decoder_hidden, encoder_outputs.transpose(0, 1).contiguous())
        enc_attn = input_mask * enc_energy + (1 - input_mask) * -1e20
        enc_attn = F.softmax(enc_attn, dim=-1)

        # batch size, graph dim
        encoder_context = torch.bmm(encoder_outputs.permute(1, 2, 0), enc_attn.unsqueeze(-1)).squeeze(2)

        combined = torch.cat([decoder_output.squeeze(0), encoder_context], dim=-1)  # batch size, decoder + graph dim
        out_emb = torch.tanh(self.pre_out(combined))  # batch size, word dim
        logits = self.out(out_emb)  # batch size, vocab size

        output = torch.zeros(batch_size, ext_vocab_size, dtype=torch.float).to(self.device)
        ptr_gen = torch.cat([x, decoder_hidden, encoder_context], dim=-1)  # batch size, word + graph + decoder * 2 dim
        prob_ptr = torch.sigmoid(self.pointer(ptr_gen))  # batch size, 1

        vocab_output = F.softmax(logits, dim=1)
        output[:, :self.word_embedding.weight.shape[0]] = vocab_output * (1 - prob_ptr)

        enc_attn2 = input_node_mask * enc_energy + (1 - input_node_mask) * -1e20  # batch size, max node size
        ptr_output = F.softmax(enc_attn2, dim=-1)[:, :encoder_word_idx.shape[1]]
        output.scatter_add_(1, encoder_word_idx, prob_ptr * ptr_output)

        return output, decoder_state, enc_attn, prob_ptr, encoder_context

    def out(self, x):
        return x.matmul(self.word_embedding.weight.transpose(0, 1))

    def enc_attn_fn(self, decoder_hidden, encoder_outputs):
        # batch size, max node size, decoder dim
        attn = self.attn_encoder_linear(encoder_outputs) + self.attn_decoder_linear(decoder_hidden).unsqueeze(1)
        return self.attn_linear(torch.tanh(attn)).squeeze(2)  # batch size, max node size

