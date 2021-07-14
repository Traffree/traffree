import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn


class GNNModel(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, hidden_dim=32, dropout=0.05):
        super(GNNModel, self).__init__()

        # dim = [num_tls, num_features]
        self.dropout = dropout
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.convs.append(self.build_conv_model(input_dim, hidden_dim))
        self.lns = nn.ModuleList()
        for _ in range(num_layers - 1):
            # TODO: ReLU ?
            self.lns.append(nn.LayerNorm(hidden_dim))
            self.convs.append(self.build_conv_model(hidden_dim, hidden_dim))

        # dim = [num_tls, hidden_dim]
        # post-message-passing
        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def build_conv_model(self, input_dim, hidden_dim):
        # refer to pytorch geometric nn module for different implementation of GNNs.
        return pyg_nn.GCNConv(input_dim, hidden_dim)

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)  # [num_tls, hidden_dim]
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if not i == self.num_layers - 1:
                x = self.lns[i](x)

        x = self.post_mp(x)
        return F.log_softmax(x, dim=1)
