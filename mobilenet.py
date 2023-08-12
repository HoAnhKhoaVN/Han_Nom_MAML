from torch.nn import Module, Linear
from torchvision.models.mobilenetv2 import mobilenet_v2

class MyMobileNetV2(Module):
    def __init__(
        self,
        num_classes: int
    ):
        super().__init__()
        self.base_model = mobilenet_v2()
        in_features = self.base_model.classifier[1].in_features
        self.base_model.classifier[1] = Linear(in_features, num_classes)
        
    def forward(self, x):
        x = self.base_model(x)
        return x