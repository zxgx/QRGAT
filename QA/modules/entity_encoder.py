import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class EntityInit(nn.Module):
    def __init__(self, relation_dim, hidden_dim, direction):
        super(EntityInit, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(relation_dim, hidden_dim),
            nn.ReLU()
        )
        self.layer_norm = nn.LayerNorm(hidden_dim * (2 if direction == 'all' else 1))
        self.direction = direction

    def forward(self, fact_relations, edge_index, num_ents):
        # fact size, hidden dim
        fact_relations = self.linear(fact_relations)

        size = list(fact_relations.shape)
        size[0] = num_ents

        if self.direction == 'outward':
            # fact size, hidden dim
            tgt_index = edge_index[1].unsqueeze(1).expand_as(fact_relations)

            # batch size * max local entity, hidden dim
            local_entity = torch.zeros(size, dtype=fact_relations.dtype, device=fact_relations.device)
            local_entity.scatter_add_(0, tgt_index, fact_relations)
            local_entity = F.relu(local_entity)
        elif self.direction == 'inward':
            # fact size, hidden dim
            src_index = edge_index[0].unsqueeze(1).expand_as(fact_relations)

            # batch size * max local entity, hidden dim
            local_entity = torch.zeros(size, dtype=fact_relations.dtype, device=fact_relations.device)
            local_entity.scatter_add_(0, src_index, fact_relations)
            local_entity = F.relu(local_entity)
        else:
            # fact size, hidden dim
            src_index = edge_index[0].unsqueeze(1).expand_as(fact_relations)
            tgt_index = edge_index[1].unsqueeze(1).expand_as(fact_relations)

            # batch size * max local entity, hidden dim
            head_ent = torch.zeros(size, dtype=fact_relations.dtype, device=fact_relations.device)
            tail_ent = torch.zeros(size, dtype=fact_relations.dtype, device=fact_relations.device)
            head_ent.scatter_add_(0, src_index, fact_relations)
            tail_ent.scatter_add_(0, tgt_index, fact_relations)

            # batch size * max local entity, hidden dim * 2
            local_entity = F.relu(torch.cat([head_ent, tail_ent], dim=1))

        return self.layer_norm(local_entity)
