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
<<<<<<< HEAD
    print(type(dataset))
=======

>>>>>>> be9c1f31270d8495ae858949dc548f4f2f27cacf
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    for v, q, a in loader:
        print(v)
        break

if __name__ == '__main__':
<<<<<<< HEAD
    main()
=======
    main()
>>>>>>> be9c1f31270d8495ae858949dc548f4f2f27cacf
