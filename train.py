from model import ViVQAModel
from data import ViVQADataset
from torch.utils.data import DataLoader
from utils import config
import pandas as pd

def main():
    # Your data loading and processing code
    df = pd.read_csv(config.__DATASET__)
    dataset = ViVQADataset(df, config.__FEATURES__)

    batch_size = config.batch_size
    print(type(dataset))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    for v, q, a in loader:
        print(v)
        break

if __name__ == '__main__':
    main()
