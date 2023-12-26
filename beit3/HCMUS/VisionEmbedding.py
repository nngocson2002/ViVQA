import timm
import torch
import torch.nn as nn
from torchvision import transforms
from lavis.models import load_model_and_preprocess
from efficientnet_pytorch import EfficientNet


class ResnetExtractor(nn.Module):
    def __init__(self):
        super(ResnetExtractor, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = timm.create_model('resnet152', pretrained=True)
        self.model.to(self.device)
        
        self.transform = self.get_transforms(target_size=224, central_fraction=0.875)
        self.pooling1 = nn.AdaptiveAvgPool2d((1, 32))
        self.pooling2 = nn.AdaptiveAvgPool2d((1, 768))
        self.model_name = 'resnet152'
        
    def get_transforms(self, target_size, central_fraction=1.0):
        return transforms.Compose([
            transforms.Resize(int(target_size / central_fraction)),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    ])
    
    def forward(self, *images):
        images_transformed = torch.stack([self.transform(image.convert('RGB')).to(self.device) for image in images])
        batch_size = images_transformed.shape[0]
        x = self.model.forward_features(images_transformed)
        x = self.pooling1(x)
        x = x.permute(0, 3, 2, 1)
        x = self.pooling2(x)
        x = x.reshape(batch_size, x.shape[1], -1)
        return x
    

class VGG16Extractor(nn.Module):
    def __init__(self):
        super(VGG16Extractor, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = timm.create_model('vgg16', pretrained=True)
        self.model.to(self.device)
        
        self.transform = self.get_transforms(target_size=224, central_fraction=0.875)
        self.pooling1 = nn.AdaptiveAvgPool2d((1, 32))
        self.pooling2 = nn.AdaptiveAvgPool2d((1, 768))
        self.model_name = 'vgg16'
        
    def get_transforms(self, target_size, central_fraction=1.0):
        return transforms.Compose([
            transforms.Resize(int(target_size / central_fraction)),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    ])
    
    def forward(self, *images):
        images_transformed = torch.stack([self.transform(image.convert('RGB')).to(self.device) for image in images])
        batch_size = images_transformed.shape[0]
        x = self.model.forward_features(images_transformed)
        x = self.pooling1(x)
        x = x.permute(0, 3, 2, 1)
        x = self.pooling2(x)
        x = x.reshape(batch_size, x.shape[1], -1)
        return x
    
class Blip2ViTExtractor(nn.Module):
    def __init__(self):
        super(Blip2ViTExtractor, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess, _ = load_model_and_preprocess(name="blip2_feature_extractor", model_type="pretrain", is_eval=True, device=self.device)
        self.preprocess = self.preprocess["eval"]
        self.model_name = "blip2"
    def forward(self, *images):
        images = torch.stack([self.preprocess(image.convert("RGB")).to(self.device) for image in images])
        image_features = self.model.extract_features(samples={"image": images}, mode="image").image_embeds
        return image_features
    
class EfficientnetExtractor(nn.Module):
    def __init__(self, model_name):
        super(EfficientnetExtractor, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = EfficientNet.from_pretrained(model_name).to(self.device)
        self.transform = self.get_transforms(target_size=224, central_fraction=0.875)
        self.pooling1 = nn.AdaptiveAvgPool2d((1, 32))
        self.pooling2 = nn.AdaptiveAvgPool2d((1, 768))
        self.model_name = model_name
    
    def get_transforms(self, target_size, central_fraction=1.0):
        return transforms.Compose([
            transforms.Resize(int(target_size / central_fraction)),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    ])

    def forward(self, *images):
        images_transformed = torch.stack([self.transform(image.convert('RGB')).to(self.device) for image in images])
        batch_size = images_transformed.size(0)
        x = self.model.extract_features(images_transformed)
        x = self.pooling1(x)
        x = x.permute(0, 3, 2, 1)
        x = self.pooling2(x)
        x = x.reshape(batch_size, x.shape[1], -1)
        return x