import timm
import torch
import torch.nn as nn
from torchvision import transforms
from lavis.models import load_model_and_preprocess
from efficientnet_pytorch import EfficientNet
from omegaconf import OmegaConf
from lavis.common.registry import registry
from transformers.utils import TensorType
from lavis.processors.base_processor import BaseProcessor


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
        self.model.eval()
        images = torch.stack([self.preprocess(image.convert("RGB")).to(self.device) for image in images])
        image_features = self.model.extract_features(samples={"image": images}, mode="image").image_embeds
        return image_features
    
class EfficientnetExtractor(nn.Module):
    def __init__(self, model_name):
        super(EfficientnetExtractor, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = EfficientNet.from_pretrained(model_name, advprop=True).to(self.device)
        self.pooling1 = nn.AdaptiveAvgPool2d((1, 32))
        self.pooling2 = nn.AdaptiveAvgPool2d((1, 768))
        self.model_name = model_name
        self.processor = self.load_preprocess()
    
    def load_preprocess(self):
        config = OmegaConf.load(registry.get_model_class(name="blip2_feature_extractor").default_config_path(model_type="pretrain"))
        preprocess_cfg = config.preprocess
        def _build_proc_from_cfg(cfg):
            return (
                registry.get_processor_class(cfg.name).from_config(cfg)
                if cfg is not None
                else BaseProcessor()
            )
        vis_proc_cfg = preprocess_cfg.get("vis_processor")
        vis_eval_cfg = vis_proc_cfg.get("eval")
        vis_processors = _build_proc_from_cfg(vis_eval_cfg)
        return vis_processors

    def forward(self, *images):
        self.model.eval()
        images_transformed = torch.stack([self.transform(image.convert('RGB')).to(self.device) for image in images])
        batch_size = images_transformed.size(0)
        x = self.model.extract_features(images_transformed)
        x = self.pooling1(x)
        x = x.permute(0, 3, 2, 1)
        x = self.pooling2(x)
        x = x.reshape(batch_size, x.shape[1], -1)
        return x
    
class Blip2EfficientExtractor(nn.Module):
    def __init__(self):
        super(Blip2EfficientExtractor, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # BLIP-2
        self.model_blip2 = registry.get_model_class(name="blip2_feature_extractor").from_pretrained(model_type="pretrain").to(self.device)
        if self.device == "cpu" or self.device == torch.device("cpu"):
            self.model_blip2 = self.model_blip2.float()
        self.model_blip2.eval()
        
        # Efficientnet
        self.model_efficientnet = EfficientNet.from_pretrained('efficientnet-b7', advprop=True).to(self.device)
        self.model_efficientnet.eval()
        self.pooling1 = nn.AdaptiveAvgPool2d((1, 32))
        self.pooling2 = nn.AdaptiveAvgPool2d((1, 768))
        
    def forward(self, images):
        # Extract global
        global_features = self.model_blip2.extract_features(samples={"image": images}, mode="image").image_embeds
        
        # Extract local
        local_features = self.model_efficientnet.extract_features(images)
        local_features = self.pooling1(local_features)
        local_features = local_features.permute(0, 3, 2, 1)
        local_features = self.pooling2(local_features)
        batch_size = images.shape[0]
        local_features = local_features.reshape(batch_size, local_features.shape[1], -1)
        
        # Combine global & local features
        v = torch.cat([global_features, local_features], dim=1)
        return v