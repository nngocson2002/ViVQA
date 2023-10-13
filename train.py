from model import ViVQAModel
from data import ViVQADataset
import torch
from torch.utils.data import DataLoader
from utils import config
import pandas as pd
import numpy as np
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

def split_dataset(df):
    train, validation, test = np.split(df.sample(frac=1, random_state=42), [int(.7*len(df)), int(.8*len(df))])

    train.reset_index(drop=True, inplace=True)
    validation.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)

    return train, validation, test

def create_loader(df):
    df_train, df_val, df_test = split_dataset(df)

    train_dataset = ViVQADataset(df_train, config.__FEATURES__)
    val_dataset = ViVQADataset(df_val, config.__FEATURES__)
    test_dataset = ViVQADataset(df_test, config.__FEATURES__)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=True
    )

    return train_loader, val_loader, test_loader

def train(model, train_loader, optimizer, criterion, epoch_idx):
    model.train()
    running_loss = 0
    last_loss = 0
    with tqdm(train_loader, 'unit_batch') as pbar:
        for i, (v, q, a) in enumerate(pbar):
            pbar.set_description(f"Epoch {epoch_idx}")
            
            # Move inputs and targets to the GPU
            v, q, a = v.to(device), q.to(device), a.to(device)
            
            optimizer.zero_grad()
            pred = model(v, q)

            # Compute the loss and its gradients
            loss = criterion(pred, a)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            running_loss += loss.item()
            
            if i % 10 == 9:
                last_loss = running_loss/10
                pbar.set_postfix(train_loss=last_loss)
                running_loss=0           
    return last_loss

def evaluate(model, val_loader, criterion):
    model.eval()
    total_eval_loss = 0
    accs = []
    with torch.no_grad():
        for i, (v, q, a) in enumerate(val_loader):
            # Move inputs and targets to the GPU
            v, q, a = v.to(device), q.to(device), a.to(device)
            pred = model(v, q)
            loss = criterion(pred, a)
            total_eval_loss += loss.item()

            indices = torch.argmax(pred, dim=1)
            total_correct = a[torch.arange(a.size(0)), indices].sum().item()
            total_samples = a.size(0)
            acc = total_correct / total_samples
            accs.append(acc)

        total_val_loss = total_eval_loss / (i+1)
        mean_acc = np.mean(accs)

    return total_val_loss, mean_acc

def main():
    # Your data loading and processing code
    df = pd.read_csv(config.__DATASET__)
    train_loader, val_loader, test_loader = create_loader(df)
    
if __name__ == '__main__':
    main()
