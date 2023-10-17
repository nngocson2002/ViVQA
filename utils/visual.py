from modules.visualEncoder import ResnetExtractor
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
        img = Image.open(os.path.join(self.path, file_name)).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return id, img
    
def create_loader(path):
    transform = get_transforms(config.image_size, config.central_fraction)
    dataset = ViQAImages(path, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=64,
        num_workers=2,
        shuffle=False
    )
    return loader
    
def extract_features(vs_encoder):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vs_encoder = vs_encoder.to(device)
    loader = create_loader(config.__IMAGES__)

    features_shape = (len(loader.dataset), config.visual_features, config.output_size, config.output_size)

    with h5py.File('features.hdf5', 'w') as f:
        features = f.create_dataset('features', shape=features_shape, dtype='float16')
        img_ids = f.create_dataset('ids', shape=(len(loader.dataset),), dtype='int32')

        i = j = 0
        with torch.no_grad():
            for ids, imgs in loader:
                imgs = Variable(imgs.cuda(non_blocking=True))
                out = vs_encoder(imgs)

                j = i + imgs.size(0)
                features[i:j, :, :] = out.data.cpu().numpy().astype('float16')
                img_ids[i:j] = np.array(ids, dtype='int32')
                i = j
            
if __name__ == '__main__':
    vs_encoder = ResnetExtractor()
    extract_features(vs_encoder)