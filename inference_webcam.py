import torch
import cv2
import time
import numpy as np
from PIL import Image
from torchvision import transforms
from models.mobilenet import DriverMonitor

LABELS = [
    'Safe Driving',           # c0
    'Texting - Right',        # c1
    'Talking Phone - Right',  # c2
    'Texting - Left',         # c3
    'Talking Phone - Left',   # c4
    'Operating Radio',        # c5
    'Drinking',               # c6
    'Reaching Behind',        # c7
    'Hair and Makeup',        # c8
    'Talking to Passenger'    # c9
]


def load_quantized_model(model_path, num_classes=10):
    print("Loading Quantized Model")
    scripted_path = model_path.replace('.pth', '_scripted.pt')
    try:
        print(f"Attempting to load scripted model from: {scripted_path}")
        model = torch.jit.load(scripted_path, map_location='cpu')
        model.eval()
        print("Scripted model loaded successfully!")
        return model
    except Exception as e:
        print(f"Could not load scripted model: {e}")
        print("Falling back to state_dict loading")
    
    try:
        model = DriverMonitor(num_classes=num_classes, pretrained=False)
        backend = 'onednn'
        torch.backends.quantized.engine = backend
        
        model.fuse_model()
        model.qconfig = torch.quantization.get_default_qconfig(backend)
        torch.quantization.prepare(model, inplace=True)
        torch.quantization.convert(model, inplace=True)
        
        state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        print("State dict loaded successfully!")
        return model
    
    except Exception as e:
        print(f"State dict loading failed: {e}")
        raise

def preprocess_frame(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform(pil_img).unsqueeze(0)

def main():
    model_path = "./checkpoints/model_quantized.pth"
    
    try:
        model = load_quantized_model(model_path, num_classes=10)
        print("Model Loaded Successfully!")

    except Exception as e:
        print(f"Error loading quantized model: {e}")
        print("\n=== Trying Float32 Model Instead ===")

        try:
            model = DriverMonitor(num_classes=10, pretrained=False)
            float_path = "./checkpoints/best_model_float32.pth"
            state_dict = torch.load(float_path, map_location='cpu')
            model.load_state_dict(state_dict)
            model.eval()
            print("Float32 model loaded successfully!")
        except Exception as e2:
            print(f"Float32 model also failed: {e2}")
            return

    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    print("Press 'q' to quit.")
    
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
    
            input_tensor = preprocess_frame(frame)
            start = time.time()
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probs, 1)
            idx = predicted_class.item()
            conf_score = confidence.item()
            end = time.time()
            fps = 1 / (end - start + 0.000001)
            label_text = f"{LABELS[idx]}: {conf_score:.2%}"
            color = (0, 255, 0) if idx == 0 else (0, 0, 255)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(frame, label_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.imshow('Driver State Monitor', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
