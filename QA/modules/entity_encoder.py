import torch
import torch.nn as nn
import torch.nn.functional as F


class EntityInit(nn.Module):
    def __init__(self, hidden_dim, direction):
        super(EntityInit, self).__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.direction = direction

    def forward(self, fact_relations, fact2head, fact2tail):
        # fact size, hidden dim
        fact_relations = self.linear(fact_relations)
        if self.direction == 'outward':
            # batch size * max local entity, hidden dim
            local_entity = F.relu(torch.sparse.mm(fact2head, fact_relations))
        elif self.direction == 'inward':
            # batch size * max local entity, hidden dim
            local_entity = F.relu(torch.sparse.mm(fact2tail, fact_relations))
        else:
            # batch size * max local entity, hidden dim
            head_ent = torch.sparse.mm(fact2head, fact_relations)
            tail_ent = torch.sparse.mm(fact2tail, fact_relations)
            # batch size * max local entity, hidden dim * 2
            local_entity = F.relu(torch.cat([head_ent, tail_ent], dim=1))
        return local_entity
