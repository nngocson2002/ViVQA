from model import ViVQAModel
from data import ViVQADataset
from torch.utils.data import DataLoader
from utils import config
import pandas as pd
import numpy as np

def compute_loss(pred, truth):
    
    pass

def main():
    # Your data loading and processing code
    df = pd.read_csv(config.__DATASET__)
    dataset = ViVQADataset(df, config.__FEATURES__)
    
    # one line split (70%, 10%, 20%)
    train, validation, test = np.split(df.sample(frac=1, random_state=42), [int(.7*len(df)), int(.8*len(df))])

    model = ViVQAModel()

    train.reset_index(drop=True, inplace=True)
    validation.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)
    dataset_train = ViVQADataset(train, config.__FEATURES__)
    dataset_vali = ViVQADataset(validation, config.__FEATURES__)
    dataset_test = ViVQADataset(test, config.__FEATURES__)
    

    batch_size = config.batch_size
    loader_train = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True
    )

    loader_vali = DataLoader(
        dataset_vali,
        batch_size=batch_size,
        shuffle=True
    )

    loader_test = DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=True
    )

    print("Train: ")
    for v, q, a in loader_train:
        pre = model(v, q)
        print(pre)
        break
    
    print("Validation: ")
    for v, q, a in loader_vali:
        pre = model(v, q)
        print(pre)
        break

    print("Test: ")
    for v, q, a in loader_test:
        pre = model(v, q)
        print(pre)
        break
if __name__ == '__main__':
    main()
