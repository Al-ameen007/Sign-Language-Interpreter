import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import time


def train_epoch(model, train_loader, optimizer, device='cpu'):
    model.train()
    total_loss = 0
    num_batches = 0
    
    for data in train_loader:
        data = data.to(device)
        
        optimizer.zero_grad()
        
        output = model(data.x, data.edge_index, data.batch)
        loss = F.cross_entropy(output, data.y)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def evaluate(model, loader, device='cpu'):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    num_batches = 0
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            
            output = model(data.x, data.edge_index, data.batch)
            loss = F.cross_entropy(output, data.y)
            
            total_loss += loss.item()
            num_batches += 1
            
            preds = output.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
    
    avg_loss = total_loss / num_batches
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy


def train_model(model, train_loader, test_loader, epochs=200, lr=0.1, device='cpu',
                lr_decay_step=40, lr_decay_gamma=0.5, weight_decay=0.0001):
    model = model.to(device)
    
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=lr,
        momentum=0.9,
        weight_decay=weight_decay,
        nesterov=True
    )
    
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=lr_decay_step, 
        gamma=lr_decay_gamma
    )
    
    history = {
        'train_losses': [],
        'test_losses': [],
        'train_accuracies': [],
        'test_accuracies': [],
        'learning_rates': []
    }
    
    best_test_acc = 0.0
    best_model_state = None
    
    print(f"\nTraining Configuration:")
    print(f"  Device: {device}")
    print(f"  Epochs: {epochs}")
    print(f"  Initial LR: {lr}")
    print(f"  LR Decay: {lr_decay_gamma} every {lr_decay_step} epochs")
    print(f"  Optimizer: SGD with Nesterov momentum")
    print("=" * 70)
    
    for epoch in range(epochs):
        start_time = time.time()
        
        current_lr = optimizer.param_groups[0]['lr']
        history['learning_rates'].append(current_lr)
        
        train_loss = train_epoch(model, train_loader, optimizer, device)
        
        train_loss_eval, train_acc = evaluate(model, train_loader, device)
        test_loss, test_acc = evaluate(model, test_loader, device)
        
        scheduler.step()
        
        history['train_losses'].append(train_loss)
        history['test_losses'].append(test_loss)
        history['train_accuracies'].append(train_acc)
        history['test_accuracies'].append(test_acc)
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model_state = model.state_dict().copy()
        
        elapsed_time = time.time() - start_time
        
        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"LR: {current_lr:.6f} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | "
              f"Time: {elapsed_time:.2f}s")
        
        if (epoch + 1) % lr_decay_step == 0 and epoch < epochs - 1:
            print(f"  → Learning rate decayed to {optimizer.param_groups[0]['lr']:.6f}")
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nBest model loaded (Test Acc: {best_test_acc:.4f})")
    
    return history


def save_model(model, path='gcn_model.pth'):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")