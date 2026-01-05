import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from data.dataset import get_dataloaders
from models.mobilenet import DriverMonitor

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    return running_loss / len(loader), 100.0 * correct / total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return running_loss / len(loader), 100.0 * correct / total

def main():
    cfg = load_config("configs/config.yaml")
    
    device = torch.device(cfg['device'] if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg['save_dir'], exist_ok=True)
    
    train_loader, val_loader = get_dataloaders(cfg)
    
    model = DriverMonitor(num_classes=cfg['num_classes']).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3)
    
    best_acc = 0.0
    
    print(f"Starting training on {device}...")
    
    for epoch in range(cfg['epochs']):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_acc)
        
        print(f"Epoch {epoch+1}/{cfg['epochs']} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(cfg['save_dir'], "best_model_float32.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model with Acc: {best_acc:.2f}%")

if __name__ == "__main__":
    main()