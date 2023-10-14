from models.resnet.model import ViVQAModel
from data import ViVQADataset
import torch
from torch.utils.data import DataLoader
from utils import config
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import os
import glob
from datetime import datetime

class ViVQATrainer():
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, epochs, save_dir=None):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.epochs = epochs

        if save_dir is not None and os.path.exists(save_dir):
            print(f"Load weight from file:{save_dir}")

            self.save_dir = save_dir
            checkpoint_path = glob.glob(f'{self.save_dir}/model_epoch*')
            if len(checkpoint_path) == 0:
                print(f'No checkpoints found in: {self.save_dir}')
                print(f'Training new model, save to: {self.save_dir}')
            else:
                self.load_check_point(checkpoint_path)
                print(f'Restore weight successful from: {checkpoint_path}')
        else:
            self.save_dir = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
            os.makedirs(self.save_dir)
            print(f'Training new model, save to: {self.save_dir}')

        self.train_loss = []
        self.val_loss = []

    def save_checkpoint(self, save_path):
        state_dict = {'model_state_dict': self.model.state_dict()}
        torch.save(state_dict, save_path)

    def load_checkpoint(self, load_path):
        state_dict = torch.load(load_path, map_location=self.device)
        self.model.load_state_dict(state_dict['model_state_dict'])

    def train(self):
        self.model.to(self.device)

        best_val_loss = 1000000
        best_val_acc = 0

        for epoch in range(1, self.epochs+1):
            with tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.epochs}', position=0) as pbar:

                self.model.train()
                train_losses = []
                total_correct = 0
                for v, q, a in pbar:   
                    v, q, a = v.to(self.device), q.to(self.device), a.to(self.device)
                    self.optimizer.zero_grad()
                    pred = self.model(v, q)
                    loss = self.criterion(pred, a)
                    loss.backward()
                    self.optimizer.step()

                    train_losses.append(loss.item())
                    indices = torch.argmax(pred, dim=1)
                    total_correct += (a == indices).sum().item()

                avg_train_loss = np.mean(train_losses)
                avg_train_acc = total_correct/len(self.train_loader.dataset)

                self.model.eval()
                val_losses = []
                total_correct = 0
                with torch.no_grad():
                    for v, q, a in self.val_loader:
                        v, q, a = v.to(self.device), q.to(self.device), a.to(self.device)
                        pred = self.model(v, q)
                        loss = self.criterion(pred, a)

                        val_losses.append(loss.item())
                        indices = torch.argmax(pred, dim=1)
                        total_correct += (a == indices).sum().item()

                avg_val_loss = np.mean(val_losses)
                avg_val_acc = total_correct/len(self.val_loader.dataset)

            print(f'train_loss={avg_train_loss:.2f}, train_accuracy={avg_train_acc*100:.2f}%, val_loss={avg_val_loss:.2f}, val_accuracy={avg_val_acc*100:.2f}%')

            if best_val_loss > avg_val_loss:
                best_val_loss = avg_val_loss
                best_val_loss_path = f"{self.save_dir}/model_best_val_loss.pt"
                self.save_checkpoint(best_val_loss_path)

            if best_val_acc < avg_val_acc:
                best_val_acc = avg_val_acc
                best_val_acc_path = f"{self.save_dir}/model_best_val_acc.pt"
                self.save_checkpoint(best_val_acc_path)
            
            model_save_path = f'{self.save_dir}/model_epoch{epoch}.pt'
            self.save_checkpoint(model_save_path)

            if os.path.exists(f'{self.save_dir}/model_epoch{epoch-1}.pt'):
                os.remove(f'{self.save_dir}/model_epoch{epoch-1}.pt')

        print("Training complete!")

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



def test(model, train_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
    model.eval()
    total_correct = 0
    for v, q, a in train_loader:
        v, q, a = v.to(device), q.to(device), a.to(device)
        pred = nn.functional.softmax(model(v, q), dim=1)
        indices = torch.argmax(pred, dim=1)
        total_correct += a[torch.arange(a.size(0)), indices].sum().item()
    
    acc = total_correct/len(train_loader.dataset)
    return acc

def main():
    df = pd.read_csv(config.__DATASET__)
    train_loader, val_loader, test_loader = create_loader(df)

    model = ViVQAModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    trainer = ViVQATrainer(model, train_loader, val_loader, optimizer, criterion, epochs=50)
    trainer.train()
    
if __name__ == '__main__':
    main()