import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GNN(nn.Module):
    def __init__(self, direction, num_layers, num_step, hidden_dim, rnn_type, dropout, kge_f):
        super(GNN, self).__init__()

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
            self.kge = self.DistMult
        elif kge_f == 'ComplEx':
            self.kge = self.ComplEx
        elif kge_f == 'TuckER':
            self.W = nn.Parameter(torch.from_numpy(np.random.uniform(-1, 1, (layer_dim, layer_dim, layer_dim))))
            self.kge = self.TuckER
        elif kge_f == 'RotatE':
            self.kge = self.RotatE
        else:
            raise ValueError('Unknown KGE func: ' + kge_f)

        for layer in range(num_layers):
            ih_name, hh_name = 'weight_ih_l{}'.format(layer), 'weight_hh_l{}'.format(layer)
            setattr(self, ih_name, nn.Linear(layer_dim, layer_dim * self.num_chunks, bias=False))
            setattr(self, hh_name, nn.Linear(layer_dim, layer_dim * self.num_chunks))

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
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

        init_kge_score = self.kge(entity_emb, head2edge, tail2edge, fact_relations)  # fact size, 1
        init_kge_loss = torch.mean(-torch.log(torch.sigmoid(init_kge_score)))
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
                # batch size, max local entity, hidden dim
                outward_neighbor = torch.sparse.mm(edge2tail, outward_prior * fact_x).view(
                    batch_size, max_local_entity, -1)
                # batch size * max local entity, 1
                outward_mask = torch.sparse.mm(edge2tail, outward_prior)

                # fact size, 1
                inward_prior = torch.sparse.mm(tail2edge, ent_label.view(-1, 1))
                # batch size, max local entity, hidden dim
                inward_neighbor = torch.sparse.mm(edge2head, inward_prior * fact_x).view(
                    batch_size, max_local_entity, -1)
                # batch size * max local entity, 1
                inward_mask = torch.sparse.mm(edge2head, inward_prior)

                # batch size, max local entity, hidden dim * 2
                neighbor = torch.cat([outward_neighbor, inward_neighbor], dim=2)
                # batch size * max local entity, 1
                inter_mask = outward_mask + inward_mask

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

    def DistMult(self, ent, head2edge, tail2edge, relation, mask=None):
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

    def ComplEx(self, ent, head2edge, tail2edge, relation, mask=None):
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
