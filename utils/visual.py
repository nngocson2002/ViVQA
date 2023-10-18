#from modules.visualEncoder import ResnetExtractor, ViTExtractor
import torch
import config
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from PIL import Image
import os
import glob
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import h5py

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
        #image = self.preprocess(x).unsqueeze(0).to(self.device)
        image_features = self.model.encode_image(x.to(self.device))
        return image_features

def get_transforms(target_size, central_fraction=1.0):
    return transforms.Compose([
        transforms.Resize(int(target_size / central_fraction)),
        transforms.CenterCrop(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

class ViQAImages(Dataset):
    def __init__(self, path, transform=None):
        super(ViQAImages, self).__init__()
        self.path = path
        self.id2filename = self.generate_id2filename()
        self.ids = list(self.id2filename.keys())
        self.transform = transform

    def generate_id2filename(self):
        d = {}
        file_paths = sorted(glob.glob(os.path.join(self.path, '*.jpg')))
        for file_path in file_paths:
            file_name = os.path.basename(file_path)
            id = file_name.split('.')[0]
            d[id] = file_name
        return d
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        id = self.ids[idx]
        file_name = self.id2filename[id]
        img = Image.open(os.path.join(self.path, file_name))#.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return id, img
    
def create_loader(path, transform):
    # transform = get_transforms(config.image_size, config.central_fraction)
    dataset = ViQAImages(path, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=64,
        num_workers=0,
        shuffle=False
    )
    return loader
    
def extract_features(vs_encoder):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = vs_encoder.preprocess
    vs_encoder = vs_encoder.to(device)
    loader = create_loader(config.__IMAGES__, transform)

    features_shape = (len(loader.dataset), 512)

    with h5py.File('features.hdf5', 'w') as f:
        features = f.create_dataset('features', shape=features_shape, dtype='float16')
        img_ids = f.create_dataset('ids', shape=(len(loader.dataset),), dtype='int32')

        i = j = 0
        with torch.no_grad():
            for ids, imgs in tqdm(loader):
                imgs = Variable(imgs.cuda(non_blocking=True))
                out = vs_encoder(imgs)

                j = i + imgs.size(0)
                features[i:j, :] = out.data.cpu().numpy().astype('float16')
                img_ids[i:j] = np.array(ids, dtype='int32')
                i = j
            
if __name__ == '__main__':
    vs_encoder = ViTExtractor()
    extract_features(vs_encoder)