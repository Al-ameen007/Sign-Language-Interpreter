import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder

from arsl.data import create_data_loaders, create_graph_data_list


def test_create_graph_data_list():
    num_samples = 3
    num_nodes = 50
    in_channels = 3

    x_list = [np.random.rand(num_nodes, in_channels) for _ in range(num_samples)]
    adj_matrices = [np.random.randint(0, 2, (num_nodes, num_nodes)) for _ in range(num_samples)]
    labels = ["label1", "label2", "label1"]

    label_encoder = LabelEncoder()
    label_encoder.fit(labels)

    data_list = create_graph_data_list(x_list, adj_matrices, labels, label_encoder)

    assert len(data_list) == num_samples
    assert data_list[0].x.shape == (num_nodes, in_channels)
    assert data_list[0].y.item() == label_encoder.transform(["label1"])[0]


def test_create_data_loaders():
    # Mocking Data objects
    from torch_geometric.data import Data

    num_samples = 4
    data_list = [
        Data(x=torch.randn(10, 3), edge_index=torch.tensor([[0, 1], [1, 0]]), y=torch.tensor([0]))
        for _ in range(num_samples)
    ]

    train_loader, test_loader = create_data_loaders(data_list, data_list, batch_size=2)

    assert len(train_loader) == 2
    assert len(test_loader) == 2

    # Check one batch
    for batch in train_loader:
        assert batch.batch.max().item() == 1  # 2 samples per batch
        break
