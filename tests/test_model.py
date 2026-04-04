import torch

from arsl.model import SpatialAttention, SpatialGCNConv, create_model


def test_spatial_attention():
    batch_size = 2
    num_nodes = 25
    in_channels = 8
    num_heads = 2

    attention = SpatialAttention(in_channels, num_heads)
    x = torch.randn(batch_size * num_nodes, in_channels)

    avg_attention = attention(x, batch_size, num_nodes)

    assert avg_attention.shape == (batch_size, num_nodes, num_nodes)
    # Check if it's a valid attention matrix (rows sum to 1)
    assert torch.allclose(avg_attention.sum(dim=-1), torch.ones(batch_size, num_nodes), atol=1e-5)


def test_spatial_gcn_conv():
    batch_size = 2
    num_nodes_per_graph = 25
    in_channels = 8
    out_channels = 16

    conv = SpatialGCNConv(in_channels, out_channels)

    x = torch.randn(batch_size * num_nodes_per_graph, in_channels)
    # Dummy edge_index (just some edges within each graph)
    edge_index = torch.tensor([[0, 1, 25, 26], [1, 0, 26, 25]], dtype=torch.long)
    batch = torch.cat([torch.zeros(num_nodes_per_graph), torch.ones(num_nodes_per_graph)]).long()

    output = conv(x, edge_index, batch)

    assert output.shape == (batch_size * num_nodes_per_graph, out_channels)


def test_model_forward():
    in_channels = 3
    hidden_channels = 8
    out_channels = 10
    num_layers = 2
    num_heads = 2

    model = create_model(in_channels, hidden_channels, out_channels, num_layers, num_heads)

    batch_size = 2
    num_frames = 4
    num_nodes_per_frame = 25
    total_nodes_per_graph = num_frames * num_nodes_per_frame

    x = torch.randn(batch_size * total_nodes_per_graph, in_channels)
    # Dummy edge_index
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    batch = torch.repeat_interleave(torch.arange(batch_size), total_nodes_per_graph)

    output = model(x, edge_index, batch)

    assert output.shape == (batch_size, out_channels)
