import torch
import torch.nn as nn

from modules import Instruction, EntityInit, GATLayer, NSM


class QAModel(nn.Module):
    def __init__(self, word_size, word_dim, hidden_dim, question_dropout, linear_dropout, num_step, pretrained_emb,
                 relation_size, relation_dim, direction, gat_head_dim, gat_head_size, gat_dropout, gat_skip, gat_bias):
        super(QAModel, self).__init__()
        assert direction in ('all', 'inward', 'outward')
        self.num_step = num_step
        self.direction = direction

        #
        # Question Encoder
        #
        if pretrained_emb is None:
            self.word_embedding = nn.Embedding(word_size, word_dim, padding_idx=0)
        else:
            self.word_embedding = nn.Embedding.from_pretrained(pretrained_emb, padding_idx=0, freeze=False)
        self.instruction_generator = Instruction(word_dim, hidden_dim, question_dropout, linear_dropout, num_step)

        #
        # Relation Embedding & Entity Encoder
        #
        self.relation_embedding = nn.Embedding(relation_size, relation_dim)
        self.entity_encoder = EntityInit(relation_dim, hidden_dim, direction)

        #
        # Graph Encoder
        #
        num_in_features = [hidden_dim * (2 if direction == 'all' else 1)] + \
                          [gat_head_dim * gat_head_size for _ in range(num_step-1)]
        num_out_features = [gat_head_dim for _ in range(num_step-1)] + [hidden_dim]
        num_heads = [gat_head_size for _ in range(num_step)]

        layers = []
        for i in range(num_step):
            layers.append(GATLayer(
                num_in_features[i], num_out_features[i], num_heads[i], relation_dim, hidden_dim,
                concat=False if i == num_step-1 else True, activation=nn.ELU() if i != num_step-1 else None,
                dropout_prob=gat_dropout, add_skip_connection=gat_skip, bias=gat_bias,
            ))
        self.gat = nn.ModuleList(layers)

    def forward(self, batch):
        question, question_mask, topic_label, entity_mask, subgraph = batch
        batch_ids, batch_relations, edge_index = subgraph

        batch_size, max_local_entity = topic_label.shape

        # batch size, max seq len, word dim
        question = self.word_embedding(question)

        # [ batch size, hidden dim ]
        # batch size, 1, hidden dim
        # [ batch size, max seq len ]
        instructions, question, attentions = self.instruction_generator(question, question_mask)

        # fact size, relation dim
        fact_relations = self.relation_embedding(batch_relations)

        # batch size * max local entity, hidden dim * num dir
        entity_emb = self.entity_encoder(fact_relations, edge_index, batch_size*max_local_entity)

        for i in range(self.num_step):
            entity_emb = self.gat[i](entity_emb, edge_index, fact_relations, instructions[i], batch_ids, max_local_entity)

        # batch size, max local entity, hidden dim
        entity_emb = entity_emb.view(batch_size, max_local_entity, -1)

        # batch size, max local entity
        predict_scores = entity_mask * question.matmul(entity_emb.transpose(1, 2)).squeeze(1) + (1 - entity_mask) * -1e20
        return predict_scores
