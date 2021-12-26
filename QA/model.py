import torch
import torch.nn as nn

from modules import Instruction, EntityInit, GATLayer, NSMLayer, GraphEncType


class QAModel(nn.Module):
    def __init__(self, word_size, word_dim, hidden_dim, question_dropout, linear_dropout, num_step, pretrained_emb,
                 entity_size, entity_dim, relation_size, relation_dim, pretrained_relation, direction, graph_encoder_type,
                 gat_head_dim, gat_head_size, gat_dropout, gat_skip, gat_bias, attn_key, attn_value):
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

        if entity_size > 0:
            self.entity_embedding = nn.Embedding(entity_size+1, entity_dim, padding_idx=entity_size)
            self.ent_linear = nn.Linear(entity_dim, hidden_dim * (2 if direction == 'all' else 1))
        else:
            self.entity_embedding = None
        #
        # Relation Embedding & Entity Encoder
        #
        if pretrained_relation is None:
            self.relation_embedding = nn.Embedding(relation_size, relation_dim)
        else:
            self.relation_embedding = nn.Embedding.from_pretrained(pretrained_relation, freeze=False)
        self.rel_norm = nn.LayerNorm(relation_dim)
        self.entity_encoder = EntityInit(relation_dim, hidden_dim, direction)

        #
        # Graph Encoder
        #
        self.graph_encoder_type = graph_encoder_type
        num_dir = 2 if direction == 'all' else 1
        layers = []
        if graph_encoder_type == GraphEncType.GAT.name:
            num_in_features = [hidden_dim * num_dir] + [gat_head_dim * gat_head_size * num_dir for _ in range(num_step-1)]
            for i in range(num_step):
                layers.append(GATLayer(
                    num_in_features[i], gat_head_dim, gat_head_size, relation_dim, hidden_dim,
                    concat=True, activation=nn.ELU(),
                    dropout_prob=gat_dropout, add_skip_connection=gat_skip, bias=gat_bias, direction=direction
                ))
            self.entity_proj = nn.Linear(gat_head_dim * gat_head_size * num_dir, hidden_dim)
        elif graph_encoder_type == GraphEncType.NSM.name:
            for i in range(num_step):
                layers.append(NSMLayer(
                    hidden_dim * num_dir, gat_head_dim, gat_head_size, hidden_dim, relation_dim, concat=True,
                    dropout=gat_dropout, direction=direction, skip=gat_skip, attn_key=attn_key, attn_value=attn_value
                ))
            self.entity_proj = nn.Linear(hidden_dim * num_dir, hidden_dim)
        elif graph_encoder_type == GraphEncType.MIX.name:
            num_in_features = [hidden_dim * num_dir] + [gat_head_dim * gat_head_size * num_dir for _ in range(num_step-1)]
            for i in range(num_step):
                layers.append(GATLayer(
                    num_in_features[i], gat_head_dim, gat_head_size, relation_dim, hidden_dim,
                    concat=True, activation=nn.ELU(),
                    dropout_prob=gat_dropout, add_skip_connection=gat_skip, bias=gat_bias, direction=direction
                ))
                layers.append(NSMLayer(
                    gat_head_dim * gat_head_size * num_dir, gat_head_dim, gat_head_size, hidden_dim, relation_dim,
                    concat=True, dropout=gat_dropout, direction=direction, skip=gat_skip
                ))
            self.entity_proj = nn.Linear(gat_head_dim * gat_head_size * num_dir, hidden_dim)
        else:
            raise ValueError("Unknown Graph Encoder Type: " + graph_encoder_type)

        self.layers = nn.ModuleList(layers)

    def forward(self, batch):
        question, question_mask, topic_label, candidate_entity, entity_mask, subgraph = batch
        batch_ids, batch_relations, edge_index = subgraph

        batch_size, max_local_entity = topic_label.shape

        # batch size, max seq len, word dim
        question = self.word_embedding(question)

        # [ batch size, hidden dim ]
        # batch size, 1, hidden dim
        # [ batch size, max seq len ]
        instructions, question, attentions = self.instruction_generator(question, question_mask)
        # print("Question: %s" % question[0, 0, :5].tolist())

        # fact size, relation dim
        fact_relations = self.rel_norm(self.relation_embedding(batch_relations))
        # print("Relation Embedding: %s" % fact_relations[0, :5].tolist())

        # batch size * max local entity, hidden dim * num dir
        if self.entity_embedding is None:
            entity_emb = self.entity_encoder(fact_relations, edge_index, batch_size*max_local_entity)
        else:
            entity_emb = self.ent_linear(self.entity_embedding(candidate_entity).view(batch_size*max_local_entity, -1))
        # print("Entity Embedding: %s" % entity_emb[0, :5].tolist())

        for i in range(self.num_step):
            # print("instruction: %s" % instructions[i][0, :5].tolist())
            if self.graph_encoder_type == GraphEncType.GAT.name:
                entity_emb = self.layers[i](
                    entity_emb, edge_index, fact_relations, instructions[i], batch_ids, max_local_entity
                )
                # print("Step %d entity embedding: %s" % (i+1, entity_emb[0, :5].tolist()))
            elif self.graph_encoder_type == GraphEncType.NSM.name:
                entity_emb, topic_label = self.layers[i](
                    entity_emb, fact_relations, instructions[i], edge_index, batch_ids, topic_label, entity_mask
                )
                # print("Step %d entity embedding: %s" % (i+1, entity_emb[0, :5].tolist()))
            elif self.graph_encoder_type == GraphEncType.MIX.name:
                entity_emb = self.layers[i*2](
                    entity_emb, edge_index, fact_relations, instructions[i], batch_ids, max_local_entity
                )
                # print("Step %d gat entity embedding: %s" % (i+1, entity_emb[0, :5].tolist()))
                entity_emb, topic_label = self.layers[i*2+1](
                    entity_emb, fact_relations, instructions[i], edge_index, batch_ids, topic_label, entity_mask
                )
                # print("Step %d nsm entity embedding: %s" % (i+1, entity_emb[0, :5].tolist()))
            else:
                ValueError("Unknown Graph Encoder Type: " + self.graph_encoder_type)

        # batch size, max local entity, hidden dim
        entity_emb = self.entity_proj(entity_emb.view(batch_size, max_local_entity, -1))
        # print("Project entity emb: %s" % entity_emb[0, 0, :5].tolist())

        # batch size, max local entity
        predict_scores = entity_mask * question.matmul(entity_emb.transpose(1, 2)).squeeze(1) + (1 - entity_mask) * -1e20
        # print("Scores: %s" % predict_scores[0, :5].tolist())
        return predict_scores
