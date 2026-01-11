import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from models.mobilenet import DriverMonitor 

def get_val_loader(data_dir, batch_size=32):
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=val_transform)
    return DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

def calibrate_model(model, loader, device):
    model.eval()
    
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            images = images.to(device)
            model(images)
            if i >= 20: 
                break

def print_size_of_model(model, label=""):
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p")
    print(f"Model: {label:<20} | Size: {size/1e6:.2f} MB")
    os.remove("temp.p")

def main():
    data_dir = "./dataset"
    checkpoint_path = "./checkpoints/best_model_float32.pth"
    quantized_save_path = "./checkpoints/model_quantized.pth"
    num_classes = 10 
    
    backend = 'onednn'
    device = torch.device('cpu') 
    
    print(f"Configuring for backend: {backend}")
    
    # Check checkpoint 
    if not os.path.exists(checkpoint_path):
        print(f"\n Error: Checkpoint not found at {checkpoint_path}")
        return
    
    # Load Float32 Model
    model = DriverMonitor(num_classes=num_classes, pretrained=False).to(device)
    
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    
    print_size_of_model(model, "Float32 (Original)")
        
    # FUSE LAYERS
    model.fuse_model()
    
    # Set QConfig
    model.qconfig = torch.quantization.get_default_qconfig(backend)
    
    # Prepare (Insert Observers)
    torch.quantization.prepare(model, inplace=True)
    
    # Calibrate
    val_loader = get_val_loader(data_dir)
    calibrate_model(model, val_loader, device)
    
    # Convert (Float32 -> Int8)
    print("Converting to quantized model...")
    torch.quantization.convert(model, inplace=True)
    
    print_size_of_model(model, "Int8 (Quantized)")
    
    torch.jit.save(torch.jit.script(model), quantized_save_path.replace('.pth', '_scripted.pt'))
    print(f"Scripted model saved to: {quantized_save_path.replace('.pth', '_scripted.pt')}")
    # Also save the regular way as backup
    torch.save(model.state_dict(), quantized_save_path)


    # Validate
    from train import validate 
    criterion = nn.CrossEntropyLoss()
    print("\nValidating Quantized Model Accuracy")
    loss, acc = validate(model, val_loader, criterion, device)
    print(f"Quantized Accuracy: {acc:.2f}%")

if __name__ == "__main__":
    main()
