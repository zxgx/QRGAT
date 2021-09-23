import torch
import torch.nn as nn

from modules import Instruction, GNN, EntityInit


class QAModel(nn.Module):
    def __init__(self, word_size, word_dim, hidden_dim, question_dropout, linear_dropout, num_step, relation_size,
                 relation_dim, direction, rnn_type, num_layers, pretrained_emb):
        super(QAModel, self).__init__()
        assert direction in ('all', 'inward', 'outward')

        if pretrained_emb is None:
            self.word_embedding = nn.Embedding(word_size, word_dim, padding_idx=0)
        else:
            self.word_embedding = nn.Embedding.from_pretrained(pretrained_emb, padding_idx=0, freeze=False)
        self.instruction_generator = Instruction(word_dim, hidden_dim, question_dropout, linear_dropout, num_step)
        self.relation_embedding = nn.Embedding(relation_size, relation_dim)
        self.relation_linear = nn.Sequential(
            nn.Linear(relation_dim, hidden_dim),
            nn.ReLU()
        )
        self.entity_encoder = EntityInit(hidden_dim, direction)
        # direction, num_layers, num_step, hidden_dim, rnn_type, dropout
        self.gnn = GNN(direction, num_layers, num_step, hidden_dim, rnn_type, linear_dropout)

    def forward(self, batch):
        question, question_mask, topic_label, entity_mask, subgraph = batch
        batch_ids, batch_relations, head2edge, tail2edge = subgraph

        batch_size, max_local_entity = topic_label.shape

        # batch size, max seq len, word dim
        question = self.word_embedding(question)

        # [ batch size, hidden dim ]
        # batch size, 1, hidden dim
        # [ batch size, max seq len ]
        instructions, question, attentions = self.instruction_generator(question, question_mask)

        # fact size, hidden dim
        fact_relations = self.relation_linear(self.relation_embedding(batch_relations))

        # batch size, max local entity, hidden dim
        entity_emb = self.entity_encoder(
            fact_relations, head2edge.transpose(0, 1), tail2edge.transpose(0, 1)
        ).view(batch_size, max_local_entity, -1)

        # [ batch size, max local entity ]
        # batch size, max local entity, hidden dim
        inter_labels, entity_emb = self.gnn(
            instructions, entity_emb, fact_relations, topic_label, entity_mask, batch_ids, head2edge, tail2edge
        )
        predict_scores = entity_mask * question.matmul(entity_emb.transpose(1, 2)).squeeze(1) + (1 - entity_mask) * -1e20
        return predict_scores
