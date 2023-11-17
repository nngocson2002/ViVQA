from models.clip_vit.vqa_bcattn import ViVQAModel
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
from transformers import get_cosine_schedule_with_warmup
import beit3_vivqa
from timm.models import create_model
import torchmetrics

class ViVQATrainer():
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, epochs, save_dir=None):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.epochs = epochs
        self.scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=len(train_loader) * int(self.epochs * 0.1),
            num_training_steps=len(train_loader) * self.epochs
        )

        if save_dir is not None and os.path.exists(save_dir):
            print(f"Load weight from file:{save_dir}")

            self.save_dir = save_dir
            checkpoint_path = glob.glob(f'{self.save_dir}/model_epoch*')
            if len(checkpoint_path) == 0:
                print(f'No checkpoints found in: {self.save_dir}')
                print(f'Training new model, save to: {self.save_dir}')
            else:
                self.load_check_point(checkpoint_path[0])
                print(f'Restore weight successful from: {checkpoint_path[0]}')
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

        accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=config.max_answers).to(self.device)
        f1_score = torchmetrics.classification.F1Score(task='multiclass', num_classes=config.max_answers).to(self.device)

        best_val_loss = 1000000
        best_val_acc = 0
        early_stopping = EarlyStopping(tolerance=5, min_delta=0.1)

        for epoch in range(1, self.epochs+1):
            with tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.epochs}', position=0) as pbar:

                self.model.train()
                train_losses = []
                for v, q, a in pbar:
                    v = v['blip'] * v['resnet']   
                    v, q, a = v.to(self.device), q.to(self.device), a.to(self.device)

                    self.optimizer.zero_grad()
                    pred = self.model(v, q['input_ids'].squeeze(dim=1), q['attention_mask'].squeeze(dim=1))
                    loss = self.criterion(pred, a)
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()

                    train_losses.append(loss.item())
                    train_acc = accuracy(pred.argmax(dim=1), a)
                    train_f1_score = f1_score(pred.argmax(dim=1), a)

                avg_train_loss = np.mean(train_losses)
                train_acc = accuracy.compute()
                train_f1_score = f1_score.compute()

                accuracy.reset()
                f1_score.reset()


                self.model.eval()
                val_losses = []
                with torch.no_grad():
                    for v, q, a in self.val_loader:
                        v = v['blip'] * v['resnet']
                        v, q, a = v.to(self.device), q.to(self.device), a.to(self.device)
                        pred = self.model(v, q['input_ids'].squeeze(dim=1), q['attention_mask'].squeeze(dim=1))
                        loss = self.criterion(pred, a)

                        val_losses.append(loss.item())
                        val_acc = accuracy(pred.argmax(dim=1), a)
                        val_f1_score = f1_score(pred.argmax(dim=1), a)

                avg_val_loss = np.mean(val_losses)
                val_acc = accuracy.compute()
                val_f1_score = f1_score.compute()

                accuracy.reset()
                f1_score.reset()

            print(f'train_loss={avg_train_loss:.2f}, train_accuracy={train_acc*100:.2f}%, train_f1={train_f1_score*100:.2f}%, val_loss={avg_val_loss:.2f}, val_accuracy={val_acc*100:.2f}%, val_f1={val_f1_score*100:.2f}%')

            if best_val_loss > avg_val_loss:
                best_val_loss = avg_val_loss
                best_val_loss_path = f"{self.save_dir}/model_best_val_loss.pt"
                self.save_checkpoint(best_val_loss_path)

            early_stopping(best_val_acc, val_acc)
            if early_stopping.early_stop:
                print("Early stopping at epoch:", epoch)
                break
            if best_val_acc < val_acc:
                best_val_acc = val_acc
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

    def __call__(self, best_val_acc, curr_val_acc):        
        if (best_val_acc - curr_val_acc) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True

def create_loader(feature_path):
    df_train = pd.read_csv(config.__DATASET_TRAIN__)
    df_val = pd.read_csv(config.__DATASET_TEST__)

    train_dataset = ViVQADataset(df_train, *feature_path)
    val_dataset = ViVQADataset(df_val, *feature_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False
    )

    return train_loader, val_loader

def main():
    feature_paths = [config.VISUAL_MODEL['Blip2-ViT']['path'], 
                     config.VISUAL_MODEL['Resnet152']['path']]
    
    train_loader, val_loader = create_loader(*feature_paths)

    model = create_model('beit3_blip2_vivqa', pretrained = False, drop_path_rate=0.5)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-5, eps=1e-8)

    trainer = ViVQATrainer(model, train_loader, val_loader, optimizer, criterion, epochs=30)
    trainer.train()
    
if __name__ == '__main__':
    main()