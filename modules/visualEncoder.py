import torch.nn as nn
from torchvision.models import resnet152
import torchvision.transforms as transforms
import clip
import torch

class ResnetExtractor(nn.Module):
    def __init__(self):
        super(ResnetExtractor, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = resnet152(pretrained=True) 
        self.transform = self.get_transforms(target_size=448, central_fraction=0.875)

        def save_output(module, input, output):
            self.buffer = output
        self.model.layer4.register_forward_hook(save_output)
        self.model_name = 'Resnet152'
    
    def get_transforms(self, target_size, central_fraction=1.0):
        return transforms.Compose([
            transforms.Resize(int(target_size / central_fraction)),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    ])

    def forward(self, *images):
        images_transformed = torch.stack([self.transform(image) for image in images])
        self.model(images_transformed)
        return self.buffer


class ViTExtractor(nn.Module):
    def __init__(self):
        super(ViTExtractor,self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load('ViT-B/32', self.device)
        self.model_name = 'CLIP-ViT'

    def forward(self, *images):
        images = torch.stack([self.preprocess(image).to(self.device) for image in images])
        image_features = self.model.encode_image(images)
        return image_features