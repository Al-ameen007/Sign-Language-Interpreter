import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from arsl.engine import evaluate, train_epoch
from arsl.model import create_model


def test_train_epoch_and_evaluate():
    in_channels = 3
    hidden_channels = 8
    out_channels = 2
    num_layers = 1
    num_heads = 1

    model = create_model(in_channels, hidden_channels, out_channels, num_layers, num_heads)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # 2 frames * 25 landmarks = 50 nodes
    num_nodes = 50
    data_list = [
        Data(x=torch.randn(num_nodes, in_channels), edge_index=torch.tensor([[0, 1], [1, 0]]), y=torch.tensor([i % 2]))
        for i in range(4)
    ]
    loader = DataLoader(data_list, batch_size=2)

    # Test train_epoch
    avg_loss = train_epoch(model, loader, optimizer)
    assert isinstance(avg_loss, float)

    # Test evaluate
    avg_loss_eval, accuracy = evaluate(model, loader)
    assert isinstance(avg_loss_eval, float)
    assert 0 <= accuracy <= 1.0
