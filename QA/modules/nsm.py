import torch
import torch.nn as nn
import numpy as np


class NSMLayer(nn.Module):
    def __init__(self, num_in_features, num_out_features, num_of_head, hidden_dim, relation_dim, concat, dropout,
                 direction, skip):
        super(NSMLayer, self).__init__()
        self.num_in_features = num_in_features
        self.num_of_head = num_of_head
        self.num_out_features = num_out_features
        self.concat = concat
        self.direction = direction
        self.skip = skip

        self.node_linear_proj = nn.Linear(num_in_features, num_of_head * num_out_features)
        self.instruction_linear_proj = nn.Linear(hidden_dim, num_of_head * num_out_features)
        self.edge_linear_proj = nn.Linear(relation_dim, num_of_head * num_out_features)

        self.prior_score_fn = nn.Parameter(torch.Tensor(1, 1, num_of_head, num_out_features))
        self.edge_score_fn = nn.Parameter(torch.Tensor(1, num_of_head, num_out_features))

        num_dir = 2 if direction == 'all' else 1
        self.weight_hh = nn.Linear(num_in_features, num_in_features * 3)
        self.weight_ih = nn.Linear(num_out_features * num_of_head * num_dir if concat else num_out_features * num_dir,
                                   num_in_features * 3)

        self.dropout = nn.Dropout(dropout)
        self.nonlinear = nn.Sequential(
            nn.Linear(2*num_out_features, num_out_features),
            nn.ReLU()
        )
        self.leakyReLU = nn.LeakyReLU(0.2)
        self.agg_norm = nn.LayerNorm(num_out_features * (num_of_head if concat else 1) * num_dir)
        self.layer_norm = nn.LayerNorm(num_in_features)
        self.score_fn = nn.Parameter(torch.Tensor(1, num_in_features))
        self.init_params()

    def init_params(self):
        nn.init.xavier_uniform_(self.instruction_linear_proj.weight)
        nn.init.xavier_uniform_(self.edge_linear_proj.weight)
        nn.init.xavier_uniform_(self.edge_score_fn)
        nn.init.xavier_uniform_(self.prior_score_fn)
        nn.init.xavier_uniform_(self.weight_ih.weight)
        nn.init.xavier_uniform_(self.weight_hh.weight)
        nn.init.xavier_uniform_(self.score_fn)

    def forward(self, nodes, edges, instruction, edge_index, batch_ids, activation, node_mask):
        batch_size, max_local_entity = activation.shape
        num_of_nodes = batch_size * max_local_entity
        src_index, trg_index = edge_index[0], edge_index[1]

        # batch size, max local entity, head size, out dim
        nodes_proj = self.node_linear_proj(nodes).view(batch_size, max_local_entity, self.num_of_head, -1)
        # batch size, head size, out dim
        instruction_proj = self.instruction_linear_proj(instruction).view(
            -1, self.num_of_head, self.num_out_features
        )

        # batch size, max local entity, head size, out dim
        node_features = nodes_proj * instruction_proj.unsqueeze(1)
        # batch size, max local entity, head size
        node_scores = self.leakyReLU(node_features * self.prior_score_fn).sum(dim=-1)
        node_scores = node_scores * activation.unsqueeze(2) + (1 - activation.unsqueeze(2)) * -1e20
        # node size, head size
        prior = torch.softmax(node_scores, dim=1).view(num_of_nodes, self.num_of_head)

        # fact size, head size, out dim
        instruction_proj = instruction_proj.index_select(0, batch_ids)

        edges_proj = self.edge_linear_proj(edges).view(-1, self.num_of_head, self.num_out_features)
        # print("projected edges: %s" % edges_proj[0, 0, :5].tolist())
        edge_features = edges_proj * instruction_proj
        # print("org features: %s" % features[0, 0, :5].tolist())
        # fact size, head size
        scores = self.leakyReLU(edge_features * self.edge_score_fn).sum(dim=-1)
        # print("org scores: %s" % scores[0, :5].tolist())
        if self.direction == 'outward':
            activation = activation.view(-1, 1)  # batch size * max local entity, 1
            # fact size, 1
            prior = activation.index_select(0, src_index)

            # batch size * max local entity
            next_activation = self.graph_aggregate(prior, trg_index, num_of_nodes).squeeze(1)
            next_activation = (next_activation.view(batch_size, max_local_entity) > 0).float()

            # fact size, head size, 1
            attentions_per_edge = self.graph_attention(scores, src_index, num_of_nodes) * prior.unsqueeze(2)
            # print("attention: %s" % attentions_per_edge[0, :5, 0].tolist())
            # fact size, head size, out dim
            weighted_features = attentions_per_edge * edge_features

            # batch size * max local entity, head size, out dim
            # These features should be very sparse
            new_features = self.graph_aggregate(weighted_features, trg_index, num_of_nodes)
            # print("new features: %s" % new_features[0, 0, :5].tolist())

        elif self.direction == 'inward':
            activation = activation.view(-1, 1)  # num nodes, 1
            prior = activation.index_select(0, trg_index)  # fact size, 1

            # batch size * max local entity
            next_activation = self.graph_aggregate(prior, src_index, num_of_nodes).squeeze(1)
            next_activation = (next_activation.view(batch_size, max_local_entity) > 0).float()

            # fact size, head size, 1
            attentions_per_edge = self.graph_attention(scores, trg_index, num_of_nodes) * prior.unsqueeze(2)
            # print("attention: %s" % attentions_per_edge[0, :5, 0].tolist())
            # fact size, head size, out dim
            weighted_features = attentions_per_edge * edge_features
            # batch size * max local entity, head size, out dim
            new_features = self.graph_aggregate(weighted_features, src_index, num_of_nodes)
            # print("new features: %s" % new_features[0, 0, :5].tolist())

        else:
            # fact size, head size, 1
            outward_prior = prior.index_select(0, src_index).unsqueeze(2)
            inward_prior = prior.index_select(0, trg_index).unsqueeze(2)

            # node size ,1
            activation = activation.view(-1, 1)
            # fact size, 1
            outward_neighbor = activation.index_select(0, src_index)
            inward_neighbor = activation.index_select(0, trg_index)
            # batch size * max local entity
            outward_activation = self.graph_aggregate(outward_neighbor, trg_index, num_of_nodes).squeeze(1)
            inward_activation = self.graph_aggregate(inward_neighbor, src_index, num_of_nodes).squeeze(1)
            # batch size, max local entity
            next_activation = ((outward_activation + inward_activation).view(batch_size, max_local_entity) > 0).float()

            # fact size, head size, 1
            outward_attention = self.graph_attention(scores, src_index, num_of_nodes) * outward_prior
            # print("outward attention: %s" % outward_attention[0, :5, 0].tolist())
            inward_attention = self.graph_attention(scores, trg_index, num_of_nodes) * inward_prior
            # print("inward attention: %s" % inward_attention[0, :5, 0].tolist())

            # node size, head size, out dim
            node_features = node_features.view(-1, self.num_of_head, self.num_out_features)
            # fact size, head size, out dim
            src_features = node_features.index_select(0, src_index)
            outward_features = self.nonlinear(torch.cat([src_features, edge_features], dim=-1))
            outward_weighted_features = outward_attention * outward_features
            trg_features = node_features.index_select(0, trg_index)
            inward_features = self.nonlinear(torch.cat([trg_features, edge_features], dim=-1))
            inward_weighted_features = inward_attention * inward_features

            # batch size * max local entity, head size, out dim
            outward_new_features = self.graph_aggregate(outward_weighted_features, trg_index, num_of_nodes)
            # print("outward new features: %s" % outward_new_features[0, 0, :5].tolist())
            inward_new_features = self.graph_aggregate(inward_weighted_features, src_index, num_of_nodes)
            # print("outward new features: %s" % inward_new_features[0, 0, :5].tolist())
            new_features = torch.cat([inward_new_features, outward_new_features], dim=2)

        if self.concat:
            last_dim = self.num_of_head * self.num_out_features * (2 if self.direction == 'all' else 1)
            new_features = new_features.view(-1, last_dim)
        else:
            new_features = new_features.mean(dim=1)
        new_features = self.agg_norm(new_features)

        out_features = self.gru_step(self.dropout(new_features), nodes)
        # print("gru output features: %s" % out_features[0, :5].tolist())

        # next_activation = next_activation * node_mask
        # next_score = (self.score_fn * self.dropout(out_features)).sum(dim=1).view(batch_size, max_local_entity)
        # next_activation = torch.softmax(next_activation * next_score + (1 - next_activation) * -1e20, dim=1)
        return out_features, next_activation

    def graph_attention(self, scores, trg_index, num_of_nodes):
        # fact size, head size
        scores = scores - scores.max()
        exp_scores = scores.exp()

        trg_index_broadcasted = self.explicit_broadcast(trg_index, exp_scores)

        size = list(scores.shape)
        size[0] = num_of_nodes
        # num nodes, head size
        neighborhood_sums = torch.zeros(size, dtype=exp_scores.dtype, device=exp_scores.device)
        neighborhood_sums.scatter_add_(0, trg_index_broadcasted, exp_scores)

        # fact size, head size
        edge_denominator = neighborhood_sums.index_select(0, trg_index)
        edge_attention = exp_scores / (edge_denominator + 1e-16)

        # fact size, head size, 1
        return edge_attention.unsqueeze(-1)

    def graph_aggregate(self, features, trg_index, num_of_nodes):
        size = list(features.shape)
        size[0] = num_of_nodes
        # num nodes, head size, out dim
        new_features = torch.zeros(size, dtype=features.dtype, device=features.device)

        trg_index_broadcasted = self.explicit_broadcast(trg_index, features)
        new_features.scatter_add_(0, trg_index_broadcasted, features)
        return new_features

    def explicit_broadcast(self, this, other):
        # Append singleton dimensions until this.dim() == other.dim()
        for _ in range(this.dim(), other.dim()):
            this = this.unsqueeze(-1)

        # Explicitly expand so that shapes are the same
        return this.expand_as(other)

    def gru_step(self, inputs, hidden):
        inputs = self.weight_ih(inputs).view(-1, 3, self.num_in_features)
        next_hidden = self.weight_hh(hidden).view(-1, 3, self.num_in_features)

        # num of nodes, dim
        update_message = torch.sigmoid(inputs[:, 0] + next_hidden[:, 0])
        reset_message = torch.sigmoid(inputs[:, 1] + next_hidden[:, 1])
        new_memory = torch.tanh(inputs[:, 2] + reset_message * next_hidden[:, 2])

        out_features = (1-update_message) * new_memory + update_message * hidden

        if self.skip:
            return self.layer_norm(out_features + hidden)
        return self.layer_norm(out_features)


class NSM(nn.Module):
    def __init__(self, direction, num_layers, num_step, hidden_dim, rnn_type, dropout, kge_f):
        super(NSM, self).__init__()

        self.direction = direction
        self.num_layers = num_layers
        self.num_step = num_step

        self.rnn_type = rnn_type.lower()
        if self.rnn_type == 'lstm':
            self.num_chunks = 4
            self.operations = ['sigmoid', 'sigmoid', 'sigmoid', 'tanh']
            self.step = self.lstm_step
        elif self.rnn_type == 'gru':
            self.num_chunks = 3
            self.operations = ['sigmoid', 'sigmoid', 'tanh']
            self.step = self.gru_step
        elif self.rnn_type == 'rnn':
            self.num_chunks = 1
            self.operations = ['tanh']
            self.step = self.rnn_step
        else:
            raise ValueError('Unknown RNN type: ' + rnn_type)

        layer_dim = hidden_dim * 2 if direction == 'all' else hidden_dim
        self.relation_linear = nn.Linear(hidden_dim, layer_dim)
        if kge_f == 'DistMult':
            self.outward_kge = self.DistMult_outward
            self.inward_kge = self.DistMult_inward
        elif kge_f == 'ComplEx':
            self.outward_kge = self.ComplEx_outward
            self.inward_kge = self.ComplEx_inward
        elif kge_f == 'TuckER':
            self.W = nn.Parameter(torch.from_numpy(np.random.uniform(-1, 1, (layer_dim, layer_dim, layer_dim))))
            self.outward_kge = self.TuckER
        elif kge_f == 'RotatE':
            self.outward_kge = self.RotatE
        else:
            raise ValueError('Unknown KGE func: ' + kge_f)

        for layer in range(num_layers):
            ih_name, hh_name = 'weight_ih_l{}'.format(layer), 'weight_hh_l{}'.format(layer)
            setattr(self, ih_name, nn.Linear(layer_dim, layer_dim * self.num_chunks, bias=False))
            setattr(self, hh_name, nn.Linear(layer_dim, layer_dim * self.num_chunks))

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.bce = nn.BCEWithLogitsLoss()
        self.layer_norm = nn.LayerNorm(layer_dim)

        self.linear_dropout = nn.Dropout(dropout)
        self.score_func = nn.Linear(layer_dim, 1)
        self.ffn = nn.Linear(layer_dim, hidden_dim)

    def forward(self, instructions, entity_emb, fact_relations, topic_label, entity_mask,
                batch_ids, head2edge, tail2edge):
        batch_size, max_local_entity = entity_emb.shape[:2]
        edge2head = head2edge.transpose(0, 1)
        edge2tail = tail2edge.transpose(0, 1)

        entity_emb = self.layer_norm(entity_emb)
        # num layers, batch size, max local entity, hidden dim * num direction
        init_state = torch.stack([entity_emb] * self.num_layers, dim=0)
        hidden_state, cell_state = init_state, init_state

        # fact size, 1
        init_outward_score = torch.sigmoid(self.outward_kge(entity_emb, head2edge, tail2edge, fact_relations)) + 1e-20
        init_inward_score = torch.sigmoid(self.inward_kge(entity_emb, head2edge, tail2edge, fact_relations)) + 1e-20
        init_kge_loss = torch.mean(-torch.log(init_outward_score)) + torch.mean(-torch.log(init_inward_score))
        kge_loss = [init_kge_loss]

        ent_label, inter_labels = topic_label, []
        for i in range(self.num_step):
            question_i = instructions[i]  # batch size, hidden dim

            # fact size, hidden dim
            fact_x = torch.index_select(question_i, dim=0, index=batch_ids)
            fact_x = self.relu(fact_x * fact_relations)

            if self.direction == 'outward':
                # fact size, 1
                fact_prior = torch.sparse.mm(head2edge, ent_label.view(-1, 1))
                # batch size, max local entity, hidden dim
                neighbor = torch.sparse.mm(edge2tail, fact_prior * fact_x).view(
                    batch_size, max_local_entity, -1)
                # batch size * max local entity, 1
                inter_mask = torch.sparse.mm(edge2tail, fact_prior)
            elif self.direction == 'inward':
                # fact size, 1
                fact_prior = torch.sparse.mm(tail2edge, ent_label.view(-1, 1))
                # batch size, max local entity, hidden dim
                neighbor = torch.sparse.mm(edge2head, fact_prior * fact_x).view(
                    batch_size, max_local_entity, -1)
                # batch size * max local entity, 1
                inter_mask = torch.sparse.mm(edge2head, fact_prior)
            else:
                # fact size, 1
                outward_prior = torch.sparse.mm(head2edge, ent_label.view(-1, 1))
                outward_support_score = self.outward_kge(hidden_state[-1], head2edge, tail2edge, fact_x)

                # batch size, max local entity, hidden dim
                outward_neighbor = torch.sparse.mm(edge2tail, outward_prior * fact_x).view(
                    batch_size, max_local_entity, -1)
                # batch size * max local entity, 1
                outward_mask = torch.sparse.mm(edge2tail, outward_prior)

                outward_kge_loss = torch.mean(-torch.log(torch.sigmoid(outward_support_score) + 1e-20))

                # fact size, 1
                inward_prior = torch.sparse.mm(tail2edge, ent_label.view(-1, 1))
                inward_support_score = self.inward_kge(hidden_state[-1], head2edge, tail2edge, fact_x)

                # batch size, max local entity, hidden dim
                inward_neighbor = torch.sparse.mm(edge2head, inward_prior * fact_x).view(
                    batch_size, max_local_entity, -1)
                # batch size * max local entity, 1
                inward_mask = torch.sparse.mm(edge2head, inward_prior)

                inward_kge_loss = torch.mean(-torch.log(torch.sigmoid(inward_support_score)+1e-20))

                # batch size, max local entity, hidden dim * 2
                neighbor = torch.cat([outward_neighbor, inward_neighbor], dim=2)
                # batch size * max local entity, 1
                inter_mask = outward_mask + inward_mask
                kge_loss.append(outward_kge_loss + inward_kge_loss)

            last_hidden = self.layer_norm(neighbor)
            hidden_splits, cell_splits = [], []
            for layer in range(self.num_layers):
                ih_name, hh_name = 'weight_ih_l{}'.format(layer), 'weight_hh_l{}'.format(layer)
                # hidden dim * num directions, hidden dim * num directions * num chunks
                weight_ih, weight_hh = getattr(self, ih_name), getattr(self, hh_name)
                hidden_split, cell_split = self.step(
                    self.linear_dropout(last_hidden), hidden_state[layer], cell_state[layer], weight_ih, weight_hh
                )
                hidden_split = self.layer_norm(hidden_split)
                cell_split = self.layer_norm(cell_split)

                hidden_splits.append(hidden_split)
                cell_splits.append(cell_split)
                last_hidden = hidden_split
            hidden_state, cell_state = torch.stack(hidden_splits, dim=0), torch.stack(cell_splits, dim=0)

            # batch size, max local entity
            inter_mask = ((inter_mask.view(batch_size, max_local_entity) + ent_label) > 1e-8).float()  # Self loop
            # inter_mask = (inter_mask > 1e-8).float().view(batch_size, max_local_entity)
            inter_mask = entity_mask * inter_mask

            inter_score = self.score_func(self.linear_dropout(last_hidden)).squeeze(2)
            inter_score = inter_mask * inter_score + (1 - inter_mask) * -1e20
            # ent_label = torch.sigmoid(inter_score)
            ent_label = torch.softmax(inter_score, dim=1)

            inter_labels.append(ent_label)
        return inter_labels, self.ffn(hidden_state[-1]), kge_loss

    def rnn_step(self, x, hidden_state, cell_state, weight_ih, weight_hh):
        op = getattr(self, self.operations[0])
        return op(weight_ih(x) + weight_hh(hidden_state)), cell_state

    def gru_step(self, x, hidden_state, cell_state, weight_ih, weight_hh):
        batch_size, max_local_entity = hidden_state.shape[:2]

        update_gate, reset_gate, memory_gate = self.operations
        update_gate = getattr(self, update_gate)
        reset_gate = getattr(self, reset_gate)
        memory_gate = getattr(self, memory_gate)

        # batch size, max local entity, num chunks, hidden dim
        x = weight_ih(x).view(batch_size, max_local_entity, self.num_chunks, -1)
        next_hidden_state = weight_hh(hidden_state).view(batch_size, max_local_entity, self.num_chunks, -1)

        # batch size, max local entity, hidden dim
        update_message = update_gate(x[:, :, 0] + next_hidden_state[:, :, 0])
        reset_message = reset_gate(x[:, :, 1] + next_hidden_state[:, :, 1])
        new_memory = memory_gate(x[:, :, 2] + reset_message * next_hidden_state[:, :, 2])
        return (1-update_message) * new_memory + update_message * hidden_state, cell_state

    def lstm_step(self, x, hidden_state, cell_state, weight_ih, weight_hh):
        batch_size, max_local_entity = hidden_state.shape[:2]

        input_gate = getattr(self, self.operations[0])
        forget_gate = getattr(self, self.operations[1])
        output_gate = getattr(self, self.operations[2])
        memory_gate = getattr(self, self.operations[3])

        # batch size, max local entity, num chunks, hidden dim
        x = (weight_ih(x) + weight_hh(hidden_state)).view(batch_size, max_local_entity, self.num_chunks, -1)

        # batch size, max local entity, hidden dim
        input_message = input_gate(x[:, :, 0])
        forget_message = forget_gate(x[:, :, 1])
        output_message = output_gate(x[:, :, 2])
        new_memory = memory_gate(x[:, :, 3])
        cell_state = forget_message * cell_state + input_message * new_memory
        hidden_state = output_message * self.tanh(cell_state)

        return hidden_state, cell_state

    def DistMult_outward(self, ent, head2edge, tail2edge, relation):
        batch_size, max_local_ent, ent_dim = ent.shape[:]
        # fact size, ent dim
        head = torch.sparse.mm(head2edge, ent.view(batch_size * max_local_ent, -1))
        tail = torch.sparse.mm(tail2edge, ent.view(batch_size * max_local_ent, -1))
        # fact size, relation dim
        relation = self.relation_linear(relation)
        relation_dim = relation.shape[1]
        assert ent_dim == relation_dim

        pred = head * relation

        score = torch.sum(pred * tail, dim=1, keepdim=True)  # fact size, 1
        return score

    def DistMult_inward(self, ent, head2edge, tail2edge, relation):
        batch_size, max_local_ent, ent_dim = ent.shape[:]
        # fact size, ent dim
        head = torch.sparse.mm(head2edge, ent.view(batch_size * max_local_ent, -1))
        tail = torch.sparse.mm(tail2edge, ent.view(batch_size * max_local_ent, -1))
        # fact size, relation dim
        relation = self.relation_linear(relation)
        relation_dim = relation.shape[1]
        assert ent_dim == relation_dim

        pred = relation * tail
        score = torch.sum(pred * head, dim=1, keepdim=True)  # fact size, 1
        return score

    def ComplEx_outward(self, ent, head2edge, tail2edge, relation):
        batch_size, max_local_ent, ent_dim = ent.shape[:]
        # fact size, ent dim
        head = torch.sparse.mm(head2edge, ent.view(batch_size * max_local_ent, -1))
        tail = torch.sparse.mm(tail2edge, ent.view(batch_size * max_local_ent, -1))
        # fact size, relation dim
        relation = self.relation_linear(relation)
        relation_dim = relation.shape[1]
        assert ent_dim == relation_dim and ent_dim % 2 == 0

        head_re, head_im = torch.chunk(head, 2, dim=1)
        relation_re, relation_im = torch.chunk(relation, 2, dim=1)

        pred_re = head_re * relation_re - head_im * relation_im
        pred_im = head_re * relation_im + head_im * relation_re
        pred = torch.cat([pred_re, pred_im], dim=1)

        score = torch.sum(pred * tail, dim=1, keepdim=True)  # fact size, 1
        return score

    def ComplEx_inward(self, ent, head2edge, tail2edge, relation):
        batch_size, max_local_ent, ent_dim = ent.shape[:]
        # fact size, ent dim
        head = torch.sparse.mm(head2edge, ent.view(batch_size * max_local_ent, -1))
        tail = torch.sparse.mm(tail2edge, ent.view(batch_size * max_local_ent, -1))
        # fact size, relation dim
        relation = self.relation_linear(relation)
        relation_dim = relation.shape[1]
        assert ent_dim == relation_dim and ent_dim % 2 == 0

        tail_re, tail_im = torch.chunk(tail, 2, dim=1)
        relation_re, relation_im = torch.chunk(relation, 2, dim=1)

        pred_re = relation_re * tail_re + relation_im * tail_im
        pred_im = relation_re * tail_im - relation_im * tail_re
        pred = torch.cat([pred_re, pred_im], dim=1)

        score = torch.sum(pred * head, dim=1, keepdim=True)  # fact size, 1
        return score

    def TuckER(self, ent, head2edge, tail2edge, relation, mask=None):
        batch_size, max_local_ent, ent_dim = ent.shape[:]
        # fact size, ent dim
        head = torch.sparse.mm(head2edge, ent.view(batch_size * max_local_ent, -1))
        tail = torch.sparse.mm(tail2edge, ent.view(batch_size * max_local_ent, -1))
        # fact size, relation dim
        relation = self.relation_linear(relation)
        relation_dim = relation.shape[1]
        assert ent_dim == relation_dim

        head = head.unsqueeze(1)  # fact size, 1, ent dim
        W_mat = torch.mm(relation, self.W.view(relation_dim, -1)).view(-1, ent_dim, ent_dim)  # fact size, ent dim, ent dim
        pred = torch.bmm(head, W_mat).squeeze(1)  # fact size, ent dim

        score = torch.sum(pred * tail, dim=1, keepdim=True)  # fact size, 1
        return score

    def RotatE(self, ent, head2edge, tail2edge, relation, mask=None):
        batch_size, max_local_ent, ent_dim = ent.shape[:]
        relation_dim = relation.shape[1]
        assert ent_dim % 2 == 0 and ent_dim // 2 == relation_dim

        # fact size, ent dim
        head = torch.sparse.mm(head2edge, ent.view(batch_size * max_local_ent, -1))
        tail = torch.sparse.mm(tail2edge, ent.view(batch_size * max_local_ent, -1))

        pi = 3.14159265358979323846
        head_re, head_im = torch.chunk(head, 2, dim=1)
        tail_re, tail_im = torch.chunk(tail, 2, dim=1)

        phase_relation = relation / (1 / ent_dim / pi)
        relation_re, relation_im = torch.cos(phase_relation), torch.sin(phase_relation)

        pred_re = head_re * relation_re - head_im * relation_im
        pred_im = head_re * relation_im + head_im * relation_re

        score = torch.cat([pred_re - tail_re, pred_im - tail_im], dim=1).norm(dim=0)

        if mask is not None:
            score *= mask

        return score
