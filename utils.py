import pandas as pd
import matplotlib.pyplot as plt


def load_data_csv(train_csv_path, test_csv_path):
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)
    
    X_train, y_train = train_df['file_path'], train_df['Label']
    X_test, y_test = test_df['file_path'], test_df['Label']
    
    return X_train, y_train, X_test, y_test


def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(history['train_losses'], label='Train Loss')
    axes[0].plot(history['test_losses'], label='Test Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Test Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(history['train_accuracies'], label='Train Accuracy')
    axes[1].plot(history['test_accuracies'], label='Test Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Test Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()


def get_device():
    import torch
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')