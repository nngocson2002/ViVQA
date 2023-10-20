from models.vit.vqa_bcattn import ViVQAModel
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
        early_stopping = EarlyStopping(tolerance=5, min_delta=0.1)

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

            early_stopping(avg_val_acc)
            if early_stopping.early_stop:
                print("Early stopping at epoch:", epoch)
                break
            if early_stopping.best_val_acc < avg_val_acc:
                best_val_acc_path = f"{self.save_dir}/model_best_val_acc.pt"
                self.save_checkpoint(best_val_acc_path)
            
            model_save_path = f'{self.save_dir}/model_epoch{epoch}.pt'
            self.save_checkpoint(model_save_path)

            if os.path.exists(f'{self.save_dir}/model_epoch{epoch-1}.pt'):
                os.remove(f'{self.save_dir}/model_epoch{epoch-1}.pt')

        print("Training complete!")

class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0):
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False
        self.best_val_acc = 0

    def __call__(self, curr_val_acc):
        if self.best_val_acc > curr_val_acc:
            self.best_val_acc = curr_val_acc
            return
        
        if (self.best_val_acc - curr_val_acc) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True

def create_loader(feature_path):
    df_train = pd.read_csv(config.__DATASET_TRAIN__)
    df_val = pd.read_csv(config.__DATASET_TEST__)

    train_dataset = ViVQADataset(df_train, feature_path)
    val_dataset = ViVQADataset(df_val, feature_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False
    )

    return train_loader, val_loader

def main():
    train_loader, val_loader = create_loader(config.VISUAL_MODEL['CLIP-ViT']['path'])
    # train_loader, val_loader = create_loader(config.VISUAL_MODEL['Resnet152']['path'])

    model = ViVQAModel(
        v_features=config.VISUAL_MODEL['CLIP-ViT']['visual_features'],
        q_features=config.TEXT_MODEL['PhoBert']['text_features'], 
        num_heads=12, 
        mid_features=config.TEXT_MODEL['PhoBert']['text_features']*2, 
        num_classes=config.max_answers, 
        num_cross_attn_layers=1, 
        dropout=0.3
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0003)

    trainer = ViVQATrainer(model, train_loader, val_loader, optimizer, criterion, epochs=50)
    trainer.train()
    
if __name__ == '__main__':
    main()