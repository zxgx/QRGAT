"""
    Inspired by https://github.com/gordicaleksa/pytorch-GAT

    Combining index_select with Tensor.scatter_add_ is more efficient than sparse matrix multiplication when
    implementing graphical softmax and aggregation
"""
import torch
import torch.nn as nn


class GATLayer(torch.nn.Module):

    src_nodes_dim = 0  # position of source nodes in edge index
    trg_nodes_dim = 1  # position of target nodes in edge index

    nodes_dim = 0      # node dimension/axis
    head_dim = 1       # attention head dimension/axis

    def __init__(self, num_in_features, num_out_features, num_of_heads, edge_dim, hidden_dim, concat, activation,
                 direction, dropout_prob=0.6, add_skip_connection=True, bias=True, log_attention_weights=False):

        super().__init__()

        # Saving these as we'll need them in forward propagation in children layers (imp1/2/3)
        self.num_of_heads = num_of_heads
        self.num_out_features = num_out_features
        self.concat = concat  # whether we should concatenate or average the attention heads
        self.add_skip_connection = add_skip_connection

        #
        # Trainable weights: linear projection matrix (denoted as "W" in the paper), attention target/source
        # (denoted as "a" in the paper) and bias (not mentioned in the paper but present in the official GAT repo)
        #

        # You can treat this one matrix as num_of_heads independent W matrices
        self.linear_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        self.linear_edge_proj = nn.Linear(edge_dim, num_of_heads * num_out_features, bias=False)

        # After we concatenate target node (node i) and source node (node j) we apply the additive scoring function
        # which gives us un-normalized score "e". Here we split the "a" vector - but the semantics remain the same.

        # Basically instead of doing [x, y] (concatenation, x/y are node feature vectors) and dot product with "a"
        # we instead do a dot product between x and "a_left" and y and "a_right" and we sum them up
        self.target_instruction = nn.Linear(hidden_dim, num_of_heads * num_out_features)
        self.source_instruction = nn.Linear(hidden_dim, num_of_heads * num_out_features)
        self.edge_instruction = nn.Linear(hidden_dim, num_of_heads * num_out_features)
        self.scoring_fn_target = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))
        self.scoring_fn_edge = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))

        # Bias is definitely not crucial to GAT - feel free to experiment (I pinged the main author, Petar, on this one)
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(num_of_heads * num_out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(num_out_features))
        else:
            self.register_parameter('bias', None)

        if add_skip_connection:
            self.skip_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        else:
            self.register_parameter('skip_proj', None)

        #
        # End of trainable weights
        #

        self.leakyReLU = nn.LeakyReLU(0.2)  # using 0.2 as in the paper, no need to expose every setting
        self.softmax = nn.Softmax(dim=-1)  # -1 stands for apply the log-softmax along the last dimension
        self.activation = activation
        # Probably not the nicest design but I use the same module in 3 locations, before/after features projection
        # and for attention coefficients. Functionality-wise it's the same as using independent modules.
        self.dropout = nn.Dropout(p=dropout_prob)

        self.log_attention_weights = log_attention_weights  # whether we should log the attention weights
        self.attention_weights = None  # for later visualization purposes, I cache the weights here

        self.init_params()
        self.direction = direction
        self.layer_norm = nn.LayerNorm(
            num_out_features * (num_of_heads if concat else 1) * (2 if direction == 'all' else 1)
        )

    def init_params(self):
        """
        The reason we're using Glorot (aka Xavier uniform) initialization is because it's a default TF initialization:
            https://stackoverflow.com/questions/37350131/what-is-the-default-variable-initializer-in-tensorflow

        The original repo was developed in TensorFlow (TF) and they used the default initialization.
        Feel free to experiment - there may be better initializations depending on your problem.

        APPEND NOTE: xavier uniform initialization is important to avoid overflow when computing edge scores
        """
        nn.init.xavier_uniform_(self.linear_proj.weight)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)
        nn.init.xavier_uniform_(self.linear_edge_proj.weight)
        nn.init.xavier_uniform_(self.source_instruction.weight)
        nn.init.xavier_uniform_(self.target_instruction.weight)
        nn.init.xavier_uniform_(self.edge_instruction.weight)
        nn.init.xavier_uniform_(self.scoring_fn_edge)

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def skip_concat_bias(self, attention_coefficients, in_nodes_features, out_nodes_features):
        if self.log_attention_weights:  # potentially log for later visualization in playground.py
            self.attention_weights = attention_coefficients

        # if the tensor is not contiguously stored in memory we'll get an error after we try to do certain ops like view
        # only imp1 will enter this one
        if not out_nodes_features.is_contiguous():
            out_nodes_features = out_nodes_features.contiguous()

        if self.add_skip_connection:  # add skip or residual connection
            if out_nodes_features.shape[-1] == in_nodes_features.shape[-1]:  # if FIN == FOUT
                # unsqueeze does this: (N, FIN) -> (N, 1, FIN), out features are (N, NH, FOUT) so 1 gets broadcast to NH
                # thus we're basically copying input vectors NH times and adding to processed vectors
                out_nodes_features += in_nodes_features.unsqueeze(1)
            else:
                # FIN != FOUT so we need to project input feature vectors into dimension that can be added to output
                # feature vectors. skip_proj adds lots of additional capacity which may cause overfitting.
                out_nodes_features += self.skip_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)

        if self.concat:
            # shape = (N, NH, FOUT) -> (N, NH*FOUT)
            out_nodes_features = out_nodes_features.view(-1, self.num_of_heads * self.num_out_features)
        else:
            # shape = (N, NH, FOUT) -> (N, FOUT)
            out_nodes_features = out_nodes_features.mean(dim=self.head_dim)

        if self.bias is not None:
            out_nodes_features += self.bias

        return out_nodes_features if self.activation is None else self.activation(out_nodes_features)

    def forward(self, in_nodes_features, edge_index, edges, instructions, batch_ids, max_local_entity):
        """
        in_nodes_features: [ batch size, feature dim ]
        edge_index: [ 2, edge size ]
        """
        src_index, trg_index = edge_index[0], edge_index[1]

        #
        # Step 1: Linear Projection + regularization
        #

        num_of_nodes = in_nodes_features.shape[self.nodes_dim]
        assert edge_index.shape[0] == 2, f'Expected edge index with shape=(2,E) got {edge_index.shape}'

        # shape = (N, FIN) where N - number of nodes in the graph, FIN - number of input features per node
        # We apply the dropout to all of the input node features (as mentioned in the paper)
        # Note: for Cora features are already super sparse so it's questionable how much this actually helps
        in_nodes_features = self.dropout(in_nodes_features)

        # shape = (N, FIN) * (FIN, NH*FOUT) -> (N, NH, FOUT) where NH - number of heads, FOUT - num of output features
        # We project the input node features into NH independent output features (one for each attention head)
        nodes_features_proj = self.linear_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)

        nodes_features_proj = self.dropout(nodes_features_proj)  # in the official GAT imp they did dropout here as well
        # print('node features: %s' % nodes_features_proj[0, 0, :5].tolist())

        # E, NH, FOUT
        edges_proj = self.dropout(self.linear_edge_proj(edges).view(-1, self.num_of_heads, self.num_out_features))
        # print('edge features: %s' % edges_proj[0, 0, :5].tolist())
        #
        # Step 2: Edge attention calculation
        #
        # N, NH, FOUT
        source_bridge = self.source_instruction(instructions).unsqueeze(1).expand(-1, max_local_entity, -1)
        source_bridge = source_bridge.contiguous().view(-1, self.num_of_heads, self.num_out_features)
        target_bridge = self.target_instruction(instructions).unsqueeze(1).expand(-1, max_local_entity, -1)
        target_bridge = target_bridge.contiguous().view(-1, self.num_of_heads, self.num_out_features)
        # E, NH, FOUT
        edge_bridge = self.edge_instruction(instructions).index_select(0, batch_ids).view(-1, self.num_of_heads, self.num_out_features)
        # print("source bridge: %s" % source_bridge[0, 0, :5].tolist())
        # print("target bridge: %s" % target_bridge[0, 0, :5].tolist())
        # print("edge bridge: %s" % edge_bridge[0, 0, :5].tolist())

        # Apply the scoring function (* represents element-wise (a.k.a. Hadamard) product)
        # shape = (N, NH, FOUT) * (1, NH, FOUT) -> (N, NH, 1) -> (N, NH) because sum squeezes the last dimension
        # Optimization note: torch.sum() is as performant as .sum() in my experiments
        scores_source = (nodes_features_proj * self.scoring_fn_source * source_bridge).sum(dim=-1)
        scores_target = (nodes_features_proj * self.scoring_fn_target * target_bridge).sum(dim=-1)
        scores_edge = (edges_proj * self.scoring_fn_edge * edge_bridge).sum(dim=-1)
        # print("source scores: %s" % scores_source[0, :5].tolist())
        # print("target scores: %s" % scores_target[0, :5].tolist())
        # print("edge scores: %s" % scores_edge[0, :5].tolist())

        # We simply copy (lift) the scores for source/target nodes based on the edge index. Instead of preparing all
        # the possible combinations of scores we just prepare those that will actually be used and those are defined
        # by the edge index.
        # scores shape = (E, NH), nodes_features_proj_lifted shape = (E, NH, FOUT), E - number of edges in the graph
        scores_source_lifted = scores_source.index_select(self.nodes_dim, src_index)
        scores_target_lifted = scores_target.index_select(self.nodes_dim, trg_index)
        scores_per_edge = self.leakyReLU(scores_source_lifted + scores_target_lifted + scores_edge)
        # print("scores: %s" % scores_per_edge[0, :5].tolist())

        if self.direction == 'outward':
            # E, NH, FOUT
            source_features_proj = nodes_features_proj.index_select(self.nodes_dim, src_index)
            # E, NH, 1
            attentions_per_edge = self.neighborhood_aware_softmax(scores_per_edge, trg_index, num_of_nodes)
            # print("Attention: %s" % attentions_per_edge[0, :5, 0].tolist())
            # E, NH, FOUT
            source_features_proj_weighted = source_features_proj * attentions_per_edge
            # N, NH, FOUT
            out_nodes_features = self.aggregate_neighbors(
                source_features_proj_weighted, trg_index, in_nodes_features, num_of_nodes
            )
            # print("aggregate features: %s" % out_nodes_features[0, 0, :5].tolist())
            out_nodes_features = self.skip_concat_bias(attentions_per_edge, in_nodes_features, out_nodes_features)
            # print("skip conn features: %s" % out_nodes_features[0, :5].tolist())

        elif self.direction == 'inward':
            # E, NH, FOUT
            target_features_proj = nodes_features_proj.index_select(self.nodes_dim, trg_index)
            # E, NH, 1
            attentions_per_edge = self.neighborhood_aware_softmax(scores_per_edge, src_index, num_of_nodes)
            # print("Attention: %s" % attentions_per_edge[0, :5, 0].tolist())
            # E, NH, FOUT
            target_features_proj_weighted = target_features_proj * attentions_per_edge
            # N, NH, FOUT
            out_nodes_features = self.aggregate_neighbors(
                target_features_proj_weighted, src_index, in_nodes_features, num_of_nodes
            )
            # print("aggregate features: %s" % out_nodes_features[0, 0, :5].tolist())
            out_nodes_features = self.skip_concat_bias(attentions_per_edge, in_nodes_features, out_nodes_features)
            # print("skip conn features: %s" % out_nodes_features[0, :5].tolist())

        else:
            # E, NH, FOUT
            source_features_proj = nodes_features_proj.index_select(self.nodes_dim, src_index)
            # E, NH, 1
            source_attention = self.neighborhood_aware_softmax(scores_per_edge, trg_index, num_of_nodes)
            # print("source attention: %s" % source_attention[0, :5, 0].tolist())
            # E, NH, FOUT
            source_features_proj_weighted = source_features_proj * source_attention
            # N, NH, FOUT
            trg_features = self.aggregate_neighbors(
                source_features_proj_weighted, trg_index, in_nodes_features, num_of_nodes
            )
            # print("target aggregate features: %s" % trg_features[0, 0, :5].tolist())
            trg_features = self.skip_concat_bias(source_attention, in_nodes_features, trg_features)
            # print("target skip conn features: %s" % trg_features[0, :5].tolist())

            target_features_proj = nodes_features_proj.index_select(self.nodes_dim, trg_index)
            target_attention = self.neighborhood_aware_softmax(scores_per_edge, src_index, num_of_nodes)
            # print("target attention: %s" % target_attention[0, :5, 0].tolist())
            target_features_proj_weighted = target_features_proj * target_attention
            src_features = self.aggregate_neighbors(
                target_features_proj_weighted, src_index, in_nodes_features, num_of_nodes
            )
            # print("source aggregate features: %s" % src_features[0, 0, :5].tolist())
            src_features = self.skip_concat_bias(target_attention, in_nodes_features, src_features)
            # print("source skip conn features: %s" % src_features[0, :5].tolist())

            # N, NH * FOUT * 2
            out_nodes_features = torch.cat([src_features, trg_features], dim=1)
        return self.layer_norm(out_nodes_features)

    #
    # Helper functions (without comments there is very little code so don't be scared!)
    #

    def neighborhood_aware_softmax(self, scores_per_edge, trg_index, num_of_nodes):
        """
        As the fn name suggest it does softmax over the neighborhoods. Example: say we have 5 nodes in a graph.
        Two of them 1, 2 are connected to node 3. If we want to calculate the representation for node 3 we should take
        into account feature vectors of 1, 2 and 3 itself. Since we have scores for edges 1-3, 2-3 and 3-3
        in scores_per_edge variable, this function will calculate attention scores like this: 1-3/(1-3+2-3+3-3)
        (where 1-3 is overloaded notation it represents the edge 1-3 and it's (exp) score) and similarly for 2-3 and 3-3
         i.e. for this neighborhood we don't care about other edge scores that include nodes 4 and 5.

        Note:
        Subtracting the max value from logits doesn't change the end result but it improves the numerical stability
        and it's a fairly common "trick" used in pretty much every deep learning framework.
        Check out this link for more details:

        https://stats.stackexchange.com/questions/338285/how-does-the-subtraction-of-the-logit-maximum-improve-learning

        """
        # Calculate the numerator. Make logits <= 0 so that e^logit <= 1 (this will improve the numerical stability)
        scores_per_edge = scores_per_edge - scores_per_edge.max()
        exp_scores_per_edge = scores_per_edge.exp()  # softmax

        # Calculate the denominator. shape = (E, NH)
        neigborhood_aware_denominator = self.sum_edge_scores_neighborhood_aware(exp_scores_per_edge, trg_index, num_of_nodes)

        # 1e-16 is theoretically not needed but is only there for numerical stability (avoid div by 0) - due to the
        # possibility of the computer rounding a very small number all the way to 0.
        attentions_per_edge = exp_scores_per_edge / (neigborhood_aware_denominator + 1e-16)

        # shape = (E, NH) -> (E, NH, 1) so that we can do element-wise multiplication with projected node features
        return attentions_per_edge.unsqueeze(-1)

    def sum_edge_scores_neighborhood_aware(self, exp_scores_per_edge, trg_index, num_of_nodes):
        # The shape must be the same as in exp_scores_per_edge (required by scatter_add_) i.e. from E -> (E, NH)
        trg_index_broadcasted = self.explicit_broadcast(trg_index, exp_scores_per_edge)

        # shape = (N, NH), where N is the number of nodes and NH the number of attention heads
        size = list(exp_scores_per_edge.shape)  # convert to list otherwise assignment is not possible
        size[self.nodes_dim] = num_of_nodes
        neighborhood_sums = torch.zeros(size, dtype=exp_scores_per_edge.dtype, device=exp_scores_per_edge.device)

        # position i will contain a sum of exp scores of all the nodes that point to the node i (as dictated by the
        # target index)
        neighborhood_sums.scatter_add_(self.nodes_dim, trg_index_broadcasted, exp_scores_per_edge)

        # Expand again so that we can use it as a softmax denominator. e.g. node i's sum will be copied to
        # all the locations where the source nodes pointed to i (as dictated by the target index)
        # shape = (N, NH) -> (E, NH)
        return neighborhood_sums.index_select(self.nodes_dim, trg_index)

    def aggregate_neighbors(self, nodes_features_proj_lifted_weighted, trg_index, in_nodes_features, num_of_nodes):
        size = list(nodes_features_proj_lifted_weighted.shape)  # convert to list otherwise assignment is not possible
        size[self.nodes_dim] = num_of_nodes  # shape = (N, NH, FOUT)
        out_nodes_features = torch.zeros(size, dtype=in_nodes_features.dtype, device=in_nodes_features.device)

        # shape = (E) -> (E, NH, FOUT)
        trg_index_broadcasted = self.explicit_broadcast(trg_index, nodes_features_proj_lifted_weighted)
        # aggregation step - we accumulate projected, weighted node features for all the attention heads
        # shape = (E, NH, FOUT) -> (N, NH, FOUT)
        out_nodes_features.scatter_add_(self.nodes_dim, trg_index_broadcasted, nodes_features_proj_lifted_weighted)

        return out_nodes_features

    def lift(self, scores_source, scores_target, nodes_features_matrix_proj, edge_index):
        """
        Lifts i.e. duplicates certain vectors depending on the edge index.
        One of the tensor dims goes from N -> E (that's where the "lift" comes from).

        """
        src_nodes_index = edge_index[self.src_nodes_dim]
        trg_nodes_index = edge_index[self.trg_nodes_dim]

        # Using index_select is faster than "normal" indexing (scores_source[src_nodes_index]) in PyTorch!
        scores_source = scores_source.index_select(self.nodes_dim, src_nodes_index)
        scores_target = scores_target.index_select(self.nodes_dim, trg_nodes_index)
        nodes_features_matrix_proj_lifted = nodes_features_matrix_proj.index_select(self.nodes_dim, src_nodes_index)

        return scores_source, scores_target, nodes_features_matrix_proj_lifted

    def explicit_broadcast(self, this, other):
        # Append singleton dimensions until this.dim() == other.dim()
        for _ in range(this.dim(), other.dim()):
            this = this.unsqueeze(-1)

        # Explicitly expand so that shapes are the same
        return this.expand_as(other)

