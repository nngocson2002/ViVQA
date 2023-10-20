import sys
sys.path.append('./')
from modules.VisualEncoder import ViTExtractor, ResnetExtractor
import torch
import config
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import os
import glob
from tqdm import tqdm
import numpy as np
import h5py

def extract_features(vs_encoder, model_name):
    file_paths = sorted(glob.glob(os.path.join(config.__IMAGES__, '*.jpg')))
    features_shape = (len(file_paths), *config.VISUAL_MODEL[model_name]['feature_shape'])

    with h5py.File(config.VISUAL_MODEL[model_name]['path'], 'w') as f:
        features = f.create_dataset('features', shape=features_shape, dtype='float16')
        img_ids = f.create_dataset('ids', shape=(len(file_paths),), dtype='int32')

        j = k = 0
        data_loader = DataLoader(file_paths, batch_size=128, shuffle=False, num_workers=4)
        for paths in tqdm(data_loader):
            ids = [os.path.basename(path).split('.')[0] for path in paths]
            imgs = [Image.open(path) for path in paths]
            
            with torch.no_grad():
                out = vs_encoder(*imgs)
                
            k = j + len(paths)
            features[j:k] = out.cpu().numpy().astype('float16')
            img_ids[j:k] = np.array(ids, dtype='int32')
            j = k
            
if __name__ == '__main__':
    vs_encoder = ResnetExtractor()
    # vs_encoder = ViTExtractor()
    extract_features(vs_encoder, model_name=vs_encoder.model_name)