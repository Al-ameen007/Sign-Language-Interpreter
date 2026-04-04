import torch
import torch.nn as nn
import torch.nn.functional as f
from torch_geometric.nn import GCNConv, global_mean_pool


class SpatialAttention(nn.Module):
    def __init__(self, in_channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.in_channels = in_channels

        self.attention_weights = nn.Parameter(torch.randn(num_heads, in_channels * 2))

    def forward(self, x, batch_size, num_nodes):
        x_reshaped = x.view(batch_size, num_nodes, self.in_channels)

        attention_scores_all_heads = []

        for head in range(self.num_heads):
            scores = torch.zeros(batch_size, num_nodes, num_nodes, device=x.device)

            for b in range(batch_size):
                x_b = x_reshaped[b]

                x_i = x_b.unsqueeze(1)
                x_j = x_b.unsqueeze(0)

                concat_features = torch.cat(
                    [
                        x_i.expand(num_nodes, num_nodes, self.in_channels),
                        x_j.expand(num_nodes, num_nodes, self.in_channels),
                    ],
                    dim=-1,
                )

                batch_scores = torch.matmul(concat_features, self.attention_weights[head])
                batch_scores = f.leaky_relu(batch_scores, negative_slope=0.2)

                scores[b] = batch_scores

            attention_scores = f.softmax(scores, dim=-1)
            attention_scores_all_heads.append(attention_scores)

        avg_attention = torch.stack(attention_scores_all_heads, dim=0).mean(dim=0)

        return avg_attention


class SpatialGCNConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=4):
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels)
        self.attention = SpatialAttention(out_channels, num_heads)

    def forward(self, x, edge_index, batch):
        x = self.conv(x, edge_index)
        x = f.relu(x)

        batch_size = batch.max().item() + 1
        num_nodes = x.size(0) // batch_size

        attention_matrix = self.attention(x, batch_size, num_nodes)

        x_reshaped = x.view(batch_size, num_nodes, -1)

        attended_x = torch.bmm(attention_matrix, x_reshaped)

        x = attended_x.view(-1, x.size(-1))

        return x


class GCNModelWithAttention(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=5, num_heads=4, dropout=0.0):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.in_channels = in_channels

        self.spatial_convs = nn.ModuleList()
        self.temporal_convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.residual_projections = nn.ModuleList()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        self.spatial_convs.append(SpatialGCNConv(in_channels, hidden_channels, num_heads))
        self.temporal_convs.append(nn.Conv1d(hidden_channels, hidden_channels, kernel_size=9, padding=4))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        self.residual_projections.append(nn.Linear(in_channels, hidden_channels))

        for _ in range(num_layers - 1):
            self.spatial_convs.append(SpatialGCNConv(hidden_channels, hidden_channels, num_heads))
            self.temporal_convs.append(nn.Conv1d(hidden_channels, hidden_channels, kernel_size=9, padding=4))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
            self.residual_projections.append(None)

        self.fc = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        batch_size = batch.max().item() + 1
        num_nodes_per_graph = x.size(0) // batch_size
        num_frames = num_nodes_per_graph // 25

        for layer_idx in range(self.num_layers):
            identity = x

            x = self.spatial_convs[layer_idx](x, edge_index, batch)

            x_temp = x.view(batch_size, 25, num_frames, -1)
            channels = x_temp.size(-1)
            x_temp = x_temp.permute(0, 1, 3, 2).contiguous()
            x_temp = x_temp.view(batch_size * 25, channels, num_frames)

            x_temp = self.temporal_convs[layer_idx](x_temp)

            x_temp = x_temp.view(batch_size, 25, channels, num_frames)
            x_temp = x_temp.permute(0, 1, 3, 2).contiguous()

            x = x_temp.reshape(-1, channels)

            x_bn = x.view(batch_size, -1, x.size(-1))
            x_bn = x_bn.permute(0, 2, 1).contiguous()
            x_bn = self.batch_norms[layer_idx](x_bn)
            x_bn = x_bn.permute(0, 2, 1).contiguous()
            x = x_bn.view(-1, x.size(-1))

            if self.residual_projections[layer_idx] is not None:
                identity = self.residual_projections[layer_idx](identity)

            x = f.relu(x + identity)

            if self.dropout is not None:
                x = self.dropout(x)

        x = global_mean_pool(x, batch)

        x = self.fc(x)

        return x


def create_model(in_channels=3, hidden_channels=16, out_channels=20, num_layers=5, num_heads=4, dropout=0.0):
    return GCNModelWithAttention(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
    )


def load_model(
    model_path, in_channels=3, hidden_channels=16, out_channels=20, num_layers=5, num_heads=4, dropout=0.0, device="cpu"
):
    model = create_model(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
    )

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    return model
