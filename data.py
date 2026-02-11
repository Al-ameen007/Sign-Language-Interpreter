import torch
import pickle
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import dense_to_sparse
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import LabelEncoder
from preprocessing import process_videos_parallel


def create_graph_data_list(X_list, adj_matrices, labels, label_encoder):
    data_list = []
    
    for X, adj_matrix, label in zip(X_list, adj_matrices, labels):
        edge_index, _ = dense_to_sparse(torch.tensor(adj_matrix, dtype=torch.float))
        edge_index = edge_index.long()
        
        x = torch.tensor(X, dtype=torch.float)
        y = torch.tensor([label_encoder.transform([label])[0]], dtype=torch.long)
        
        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)
    
    return data_list


def prepare_dataset(file_paths, labels, label_encoder=None, max_workers=4):
    if label_encoder is None:
        label_encoder = LabelEncoder()
        label_encoder.fit(labels)
    
    processed_data = process_videos_parallel(file_paths, labels, max_workers)
    
    X_list = [item[0] for item in processed_data]
    adj_matrices = [item[1] for item in processed_data]
    labels_list = [item[2] for item in processed_data]
    
    data_list = create_graph_data_list(X_list, adj_matrices, labels_list, label_encoder)
    
    return data_list, label_encoder


def save_dataset(data_list, label_encoder, train_path='train_data.pth', 
                test_path=None, encoder_path='label_encoder.pkl', is_train=True):
    if is_train:
        torch.save(data_list, train_path)
    else:
        torch.save(data_list, test_path if test_path else 'test_data.pth')
    
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)


def load_dataset(train_path='train_data.pth', test_path='test_data.pth', 
                encoder_path='label_encoder.pkl'):
    train_data = torch.load(train_path)
    test_data = torch.load(test_path)
    
    with open(encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    
    return train_data, test_data, label_encoder


def create_data_loaders(train_data, test_data, batch_size=32, shuffle=True):
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader