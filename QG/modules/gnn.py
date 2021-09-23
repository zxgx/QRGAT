import torch
import torch.nn as nn
import torch.nn.functional as F


class GNN(nn.Module):
    def __init__(self, hops, direction, levi_graph, graph_dim):
        super(GNN, self).__init__()
        self.hops = hops
        self.direction = direction
        assert direction in ('all', 'inward', 'outward')

        self.levi_graph = levi_graph
        if not levi_graph:
            self.linear_fuse = nn.Linear(graph_dim * 2, graph_dim)

        if direction == 'all':
            self.ffn_z = nn.Linear(4 * graph_dim, graph_dim)

        self.update_gate = nn.Linear(graph_dim*2, graph_dim, bias=False)
        self.reset_gate = nn.Linear(graph_dim*2, graph_dim, bias=False)
        self.memory_gate = nn.Linear(graph_dim*2, graph_dim, bias=False)

        self.linear_max = nn.Linear(graph_dim, graph_dim, bias=False)

    def forward(self, node_vec, edge_vec, node2edge, edge2node, node_mask):
        for _ in range(self.hops):
            if self.direction == 'inward':
                # batch size, virtual node size, graph dim
                new_vec = self.gnn_step(node_vec, edge_vec, edge2node.transpose(1, 2), node2edge.transpose(1, 2))
            elif self.direction == 'outward':
                # batch size, virtual node size, graph dim
                new_vec = self.gnn_step(node_vec, edge_vec, node2edge, edge2node)
            else:
                out_vec = self.gnn_step(node_vec, edge_vec, node2edge, edge2node)
                in_vec = self.gnn_step(node_vec, edge_vec, edge2node.transpose(1, 2), node2edge.transpose(1, 2))
                new_vec = self.gated_fusion(in_vec, out_vec)  # batch size, virtual node size, graph dim
            node_vec = self.gru_step(node_vec, new_vec)  # batch size, virtual node size, graph dim

        graph_embedding = self.graph_maxpooling(node_vec, node_mask)  # batch size, graph dim
        return node_vec.transpose(0, 1), graph_embedding.unsqueeze(0)

    def gnn_step(self, node_vec, edge_vec, node2edge, edge2node):
        edge_emb = torch.bmm(node2edge, node_vec)  # batch size, edge size, graph dim
        if not self.levi_graph:
            edge_emb = torch.relu(self.linear_fuse(torch.cat([edge_emb, edge_vec], dim=-1)))

        norm = torch.sum(edge2node, 2, keepdim=True) + 1  # batch size, virtual node size, 1
        new_vec = (torch.bmm(edge2node, edge_emb) + node_vec) / norm  # batch size, node size, graph dim
        return new_vec

    def gated_fusion(self, in_vec, out_vec):
        z = torch.sigmoid(self.ffn_z(torch.cat([in_vec, out_vec, in_vec*out_vec, in_vec-out_vec], dim=-1)))
        ret = (1 - z) * in_vec + z * out_vec
        return ret

    def gru_step(self, cur, new):
        update_message = torch.sigmoid(self.update_gate(torch.cat([cur, new], dim=-1)))
        reset_message = torch.sigmoid(self.reset_gate(torch.cat([cur, new], dim=-1)))
        new_memory = torch.tanh(self.memory_gate(torch.cat([reset_message * cur, new], dim=-1)))
        ret = (1 - update_message) * cur + update_message * new_memory
        return ret

    def graph_maxpooling(self, node_state, node_mask=None):
        # Maxpool
        # Shape: (batch_size, hidden_size, num_entities)
        node_embedding_p = self.linear_max(node_state).transpose(-1, -2)
        graph_embedding = F.max_pool1d(node_embedding_p, kernel_size=node_embedding_p.size(-1)).squeeze(-1)
        return graph_embedding
