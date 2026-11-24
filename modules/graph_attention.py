import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GraphAttentionLayer(nn.Module):
    """
    Single graph attention layer.
    Propagates information among disease nodes to capture dependencies between diseases.
    """
    def __init__(self, in_features, out_features, num_heads=4, dropout=0.1, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.dropout = dropout
        self.concat = concat

        assert out_features % num_heads == 0, "out_features must be divisible by num_heads"
        self.out_dim = out_features // num_heads

        # Linear transformation matrix W
        self.W = nn.Parameter(torch.Tensor(in_features, out_features))

        # Attention parameters a_src and a_dst
        self.a_src = nn.Parameter(torch.Tensor(num_heads, self.out_dim))
        self.a_dst = nn.Parameter(torch.Tensor(num_heads, self.out_dim))

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout_layer = nn.Dropout(dropout)

        self._init_parameters()

    def _init_parameters(self):
        """Initialize parameters."""
        stdv = 1. / math.sqrt(self.out_features)
        self.W.data.uniform_(-stdv, stdv)
        self.a_src.data.uniform_(-stdv, stdv)
        self.a_dst.data.uniform_(-stdv, stdv)

    def forward(self, h, adj_matrix):
        """
        Args:
            h: node features [B, N, in_features], N = 14 diseases
            adj_matrix: adjacency matrix [B, N, N] or [N, N]

        Returns:
            h_new: updated node features [B, N, out_features] (concat=True or concat=False)
        """
        B, N, _ = h.shape

        # Linear transformation: Wh
        Wh = torch.matmul(h, self.W)  # [B, N, out_features]
        Wh = Wh.view(B, N, self.num_heads, self.out_dim)  # [B, N, num_heads, out_dim]
        Wh = Wh.permute(0, 2, 1, 3)  # [B, num_heads, N, out_dim]

        # Compute attention coefficients
        # e_ij = LeakyReLU(a_src^T Wh_i + a_dst^T Wh_j)
        Wh_src = Wh  # [B, num_heads, N, out_dim]
        Wh_dst = Wh  # [B, num_heads, N, out_dim]

        # Attention scores for source and destination nodes
        e_src = torch.matmul(Wh_src, self.a_src.unsqueeze(-1)).squeeze(-1)  # [B, num_heads, N]
        e_dst = torch.matmul(Wh_dst, self.a_dst.unsqueeze(-1)).squeeze(-1)  # [B, num_heads, N]

        # Broadcast to compute attention scores for all node pairs
        e = e_src.unsqueeze(-1) + e_dst.unsqueeze(-2)  # [B, num_heads, N, N]
        e = self.leaky_relu(e)

        # Handle the adjacency matrix (supports batched and single matrices)
        if adj_matrix.dim() == 2:
            adj_matrix = adj_matrix.unsqueeze(0).unsqueeze(0)  # [1, 1, N, N]
        elif adj_matrix.dim() == 3:
            adj_matrix = adj_matrix.unsqueeze(1)  # [B, 1, N, N]

        # Keep only the edges marked as 1 in the adjacency matrix
        e = e * adj_matrix + (-1e9) * (1 - adj_matrix)  # [B, num_heads, N, N]

        # Softmax normalization
        alpha = F.softmax(e, dim=-1)  # [B, num_heads, N, N]
        alpha = self.dropout_layer(alpha)

        # Weighted sum
        h_new = torch.matmul(alpha, Wh)  # [B, num_heads, N, out_dim]
        h_new = h_new.permute(0, 2, 1, 3).contiguous()  # [B, N, num_heads, out_dim]
        h_new = h_new.view(B, N, -1)  # [B, N, num_heads * out_dim]

        if self.concat:
            return F.elu(h_new)
        else:
            return h_new


class DiseaseDependencyGAT(nn.Module):
    """
    Disease Dependency Graph Attention Network (DD-GAT).

    Motivation: strong co-occurrence relationships exist between diseases;
    modeling these dependencies improves label prediction and knowledge retrieval.

    Method:
    - Builds a learnable disease adjacency matrix A ∈ R^{14x14}, constrained to [0, 1] by sigmoid
    - Embeds each label prediction into a d_model dimensional space
    - Propagates information among disease nodes with L layers of multi-head GAT
    - Outputs graph-enhanced label representations [B, 14, d_model] that replace the
      simple label embedding used to query the knowledge base in the original method
    """
    def __init__(self, num_labels=14, d_model=512, num_gat_layers=2, num_gat_heads=4, dropout=0.1):
        super(DiseaseDependencyGAT, self).__init__()
        self.num_labels = num_labels
        self.d_model = d_model
        self.num_gat_layers = num_gat_layers
        self.num_gat_heads = num_gat_heads

        # Learnable adjacency matrix (unnormalized)
        self.adj_logits = nn.Parameter(torch.Tensor(num_labels, num_labels))

        # Learnable self-loop strength (each disease node learns its own)
        # initialized to 0.5 (sigmoid(0)=0.5): the model decides how much
        # of its own information to keep
        self.self_loop_logits = nn.Parameter(torch.zeros(num_labels))

        # Embedding layer from label values to d_model dimensions
        self.label_embed = nn.Linear(1, d_model)

        # Multi-layer GAT
        gat_layers = []
        for i in range(num_gat_layers):
            in_dim = d_model if i == 0 else d_model
            out_dim = d_model
            concat = (i < num_gat_layers - 1)  # the last layer does not concat

            gat_layers.append(
                GraphAttentionLayer(
                    in_features=in_dim,
                    out_features=out_dim,
                    num_heads=num_gat_heads,
                    dropout=dropout,
                    concat=concat
                )
            )

        self.gat_layers = nn.ModuleList(gat_layers)
        self.dropout = nn.Dropout(dropout)

        self._init_parameters()

    def _init_parameters(self):
        """Initialize the adjacency matrix parameters."""
        # Initialize with small random values so that not all edges are active initially
        nn.init.xavier_uniform_(self.adj_logits)

    def forward(self, vis_labels):
        """
        Args:
            vis_labels: visual label predictions [B, 14]

        Returns:
            graph_query: graph-enhanced label representations [B, 14, d_model]
        """
        B = vis_labels.shape[0]

        # Compute the adjacency matrix (sigmoid constraint + symmetrization + learnable self-loops)
        adj_raw = torch.sigmoid(self.adj_logits)  # [14, 14]
        adj_sym = (adj_raw + adj_raw.t()) / 2  # symmetrize

        # Learnable self-loops: each disease node independently controls
        # how much of its own information is kept; sigmoid constrains to [0, 1],
        # initial value is about 0.5
        self_loop = torch.sigmoid(self.self_loop_logits)  # [14]
        adj_matrix = adj_sym + torch.diag(self_loop)  # [14, 14]

        # Embed label values into the d_model dimensional space
        h = self.label_embed(vis_labels.unsqueeze(-1))  # [B, 14, d_model]
        h = self.dropout(h)

        # Propagate information through the GAT layers
        for gat_layer in self.gat_layers:
            h = gat_layer(h, adj_matrix)
            h = self.dropout(h)

        graph_query = h  # [B, 14, d_model]
        return graph_query

    def get_adj_matrix(self):
        """Return the learned adjacency matrix (for visualization)."""
        return torch.sigmoid(self.adj_logits).detach().cpu().numpy()
