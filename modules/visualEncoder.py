import torch.nn as nn
from torchvision.models import resnet152
import torchvision.transforms as transforms
import clip
from lavis.models import load_model_and_preprocess
import torch

class ResnetExtractor(nn.Module):
    def __init__(self):
        super(ResnetExtractor, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = resnet152().to(self.device)
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
        images_transformed = torch.stack([self.transform(image.convert('RGB')).to(self.device) for image in images])
        self.model(images_transformed)
        return self.buffer

class ClipViTExtractor(nn.Module):
    def __init__(self):
        super(ClipViTExtractor,self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load('ViT-B/32', self.device)
        self.model_name = 'CLIP-ViT'

    def forward(self, *images):
        images = torch.stack([self.preprocess(image).to(self.device) for image in images])
        image_features = self.model.encode_image(images)
        return image_features
    
class Blip2ViTExtractor(nn.Module):
    def __init__(self):
        super(Blip2ViTExtractor, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess, _ = load_model_and_preprocess(name="blip2_feature_extractor", model_type="pretrain", is_eval=True, device=self.device)
        self.preprocess = self.preprocess["eval"]
        self.model_name = "Blip2-ViT"
    def forward(self, *images):
        images = torch.stack([self.preprocess(image.convert("RGB")).to(self.device) for image in images])
        image_features = self.model.extract_features(samples={"image": images}, mode="image").image_embeds_proj[:,0,:]
        return image_features