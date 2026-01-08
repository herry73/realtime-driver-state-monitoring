import torch
import torch.nn as nn
from torchvision.models.quantization import mobilenet_v3_large
from torchvision.models import MobileNet_V3_Large_Weights

class DriverMonitor(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        
        if pretrained:
            weights = MobileNet_V3_Large_Weights.DEFAULT
        else:
            weights = None
        # quantize=False gives us quantizable structure with FP32 weights
        self.backbone = mobilenet_v3_large(weights=weights, quantize=False)
        
        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        return x
    def fuse_model(self):
        self.backbone.fuse_model()
