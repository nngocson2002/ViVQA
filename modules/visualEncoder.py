import torch.nn as nn
from torchvision.models import resnet152
import clip
import torch
from PIL import Image


class ResnetExtractor(nn.Module):
    def __init__(self):
        super(ResnetExtractor, self).__init__()
        self.model = resnet152()

        def save_output(module, input, output):
            self.buffer = output
        self.model.layer4.register_forward_hook(save_output)

    def forward(self, x):
        self.model(x)
        return self.buffer

class ViTExtractor(nn.Module):
    def __init__(self):
        super(ViTExtractor,self).__init__()
        # Load the model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load('ViT-B/32', self.device)

    def forward(self, x):
        image = self.preprocess(Image.open(x)).unsqueeze(0).to(self.device)
        image_features = self.model.encode_image(image)
        return image_features