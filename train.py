from model import ViVQAModel
from data import ViVQADataset
from torch.utils.data import DataLoader
from utils import config
import pandas as pd

def compute_loss(pred, truth):
    
    pass

def main():
    # Your data loading and processing code
    df = pd.read_csv(config.__DATASET__)
    dataset = ViVQADataset(df, config.__FEATURES__)
    model = ViVQAModel()

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0
    )

    for v, q, a in loader:
        pred = model(v, q)
        print(pred)
        pass

if __name__ == '__main__':
    main()
