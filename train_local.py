from __future__ import division
from __future__ import print_function

import os
import glob
import time
import random
import argparse
import numpy as np
import torch
from collections import defaultdict
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from sklearn.metrics import f1_score, matthews_corrcoef, confusion_matrix, classification_report
import pickle
import json
from datetime import datetime
import torch.cuda.amp as amp
from torch.utils.checkpoint import checkpoint

from utils import load_data, accuracy
from model import GAT

class WeightedTemporalLoss(nn.Module):
    def __init__(self, alpha=0.5, temperature=1.0, device=None):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.device = device
        
    def forward(self, predictions, targets, tweet_counts, price_changes):
        # Ensure proper dimensions and broadcasting
        batch_size = predictions.size(0)
        
        # Normalize weights per batch
        tweet_weights = 1.0 / (tweet_counts + 1)  # [batch_size]
        tweet_weights = tweet_weights / tweet_weights.sum()
        
        # Reduce price_changes from [batch_size, 3] to [batch_size] by taking mean or max
        price_changes = price_changes.abs().mean(dim=1)  # or use .max(dim=1)[0]
        price_weights = price_changes  # [batch_size]
        price_weights = price_weights / (price_weights.sum() + 1e-8)  # Add epsilon to prevent division by zero
        
        # Ensure both weights have the same shape [batch_size]
        tweet_weights = tweet_weights.view(-1)
        price_weights = price_weights.view(-1)
        
        # Combine weights
        weights = self.alpha * tweet_weights + (1 - self.alpha) * price_weights
        
        # Temperature scaling
        predictions = predictions / self.temperature
        
        # Calculate loss
        loss = F.cross_entropy(predictions, targets, reduction='none')
        
        # Ensure weights match loss dimension
        weights = weights.to(loss.device)
        
        return (loss * weights).sum()

class ImprovedTrainer:
    def __init__(self, model, optimizer, scheduler, criterion, device, max_grad_norm=1.0):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.max_grad_norm = max_grad_norm
        self.device = device
        
        # Only use scaler if using CUDA
        self.use_amp = isinstance(device, torch.device) and device.type == 'cuda'
        self.scaler = amp.GradScaler() if self.use_amp else None
        
    def train_step(self, text_input, price_input, adj, targets, tweet_counts, price_changes):
        self.model.train()
        self.optimizer.zero_grad()
        
        if self.use_amp:
            with amp.autocast():
                outputs = self.model(text_input, price_input, adj)
                loss = self.criterion(outputs, targets, tweet_counts, price_changes)
            
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            outputs = self.model(text_input, price_input, adj)
            loss = self.criterion(outputs, targets, tweet_counts, price_changes)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
        self.scheduler.step(loss)
        return loss.item(), outputs

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--fastmode', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=14)
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--nb_heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.38)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--continue_epochs', type=int, default=None)
    return parser.parse_args()

def setup_environment(args):
    # Check if MPS is available
    if torch.backends.mps.is_available():
        args.device = torch.device("mps")
    elif torch.cuda.is_available() and not args.no_cuda:
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def load_checkpoint(checkpoint_path, model, stock_num, args):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    criterion = WeightedTemporalLoss(alpha=0.5, temperature=1.0)
    optimizer = optim.Adam(model.parameters(),
                          lr=checkpoint['args']['lr'],
                          weight_decay=checkpoint['args']['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10)
    
    if args.cuda:
        model = model.cuda()
        criterion = criterion.cuda()
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    criterion.load_state_dict(checkpoint['criterion_state_dict'])
    
    trainer = ImprovedTrainer(model, optimizer, scheduler, criterion)
    
    training_log = {
        'epochs': [],
        'train_loss': [],
        'train_acc': [],
        'best_f1': checkpoint.get('f1_score', 0.0),
        'best_mcc': checkpoint.get('mcc', 0.0),
        'hyperparameters': checkpoint['args']
    }
    
    return trainer, checkpoint['epoch'], training_log

def train(epoch, trainer, data_paths, adj, max_tweet_param, device):
    i = epoch % 394
    
    # Load data and move to device
    train_text = torch.tensor(
        np.load(f"{data_paths['train_text']}{str(i).zfill(10)}.npy")[:, :, :max_tweet_param, :],
        dtype=torch.float32
    ).to(device)
    
    train_price = torch.tensor(
        np.load(f"{data_paths['train_price']}{str(i).zfill(10)}.npy"),
        dtype=torch.float32
    ).to(device)
    
    train_label = torch.LongTensor(
        np.load(f"{data_paths['train_label']}{str(i).zfill(10)}.npy")
    ).to(device)
    
    # Calculate tweet counts per sample
    tweet_counts = (train_text.sum(dim=-1) != 0).float().sum(dim=-1).sum(dim=-1)  # [batch_size]
    
    # Calculate price changes per sample
    price_changes = train_price[:, -1] - train_price[:, 0]  # [batch_size]
    
    loss, output = trainer.train_step(
        train_text, train_price, adj,
        torch.max(train_label,1)[1],
        tweet_counts, price_changes
    )
    
    acc_train = accuracy(output, torch.max(train_label,1)[1])
    
    print(f"Epoch {epoch}, batch {i}, Loss: {loss}, Accuracy: {acc_train.item()}")
    return loss, acc_train.item()

def test_dict(model, adj, data_paths, max_tweet_param):
    pred_dict = {}
    with open('label_data.p', 'rb') as fp:
        true_label = pickle.load(fp)
    with open('price_feature_data.p', 'rb') as fp:
        feature_data = pickle.load(fp)
    with open('text_feature_data.p', 'rb') as fp:
        text_ft_data = pickle.load(fp)
        
    model.eval()
    test_metrics = defaultdict(list)
    
    for dates in feature_data.keys():
        with torch.no_grad():
            test_text = torch.tensor(text_ft_data[dates][:, :, :max_tweet_param, :], dtype=torch.float32).cuda()
            test_price = torch.tensor(feature_data[dates], dtype=torch.float32).cuda()
            test_label = torch.LongTensor(true_label[dates]).cuda()
            
            output = model(test_text, test_price, adj)
            output = F.softmax(output, dim=1)
            
            pred_dict[dates] = output.cpu().detach().numpy()
            loss = F.nll_loss(output, torch.max(test_label,1)[0])
            acc = accuracy(output, torch.max(test_label,1)[1])
            
            preds = torch.max(output,1)[1].cpu().numpy()
            trues = torch.max(test_label,1)[1].cpu().numpy()
            
            test_metrics['preds'].extend(preds)
            test_metrics['trues'].extend(trues)
            test_metrics['loss'].append(loss.item())
            test_metrics['acc'].append(acc.item())
    
    # Calculate final metrics
    f1 = f1_score(test_metrics['trues'], test_metrics['preds'], average='micro')
    mcc = matthews_corrcoef(test_metrics['trues'], test_metrics['preds'])
    
    print(f"Test results: loss={np.mean(test_metrics['loss']):.4f}, "
          f"accuracy={np.mean(test_metrics['acc']):.4f}, "
          f"F1={f1:.4f}, MCC={mcc:.4f}")
    
    with open('pred_dict.p', 'wb') as fp:
        pickle.dump(pred_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
    
    return f1, mcc

def main():
    args = parse_args()
    setup_environment(args)
    
    # Load data
    adj = load_data()
    stock_num = adj.size(0)
    
    # Setup paths
    data_paths = {
        'train_text': "train_text/",
        'train_price': "train_price/",
        'train_label': "train_label/",
        'val_text': "val_text/",
        'val_price': "val_price/",
        'val_label': "val_label/",
        'test_text': "test_text/",
        'test_price': "test_price/",
        'test_label': "test_label/"
    }
    
    # Create directories
    os.makedirs('model_checkpoints', exist_ok=True)
    os.makedirs('training_logs', exist_ok=True)
    
    # Initialize model and training components
    model = GAT(nfeat=64, nhid=args.hidden, nclass=2, dropout=args.dropout,
                nheads=args.nb_heads, alpha=args.alpha, stock_num=stock_num).to(torch.float32)
    
    if args.resume:
        trainer, start_epoch, training_log = load_checkpoint(args.resume, model, stock_num, args)
    else:
        criterion = WeightedTemporalLoss(alpha=0.5, temperature=1.0, device=args.device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10)
        
        model = model.to(args.device).to(torch.float32)
        criterion = criterion.to(args.device)
        adj = adj.to(args.device).to(torch.float32)
        
        trainer = ImprovedTrainer(model, optimizer, scheduler, criterion, args.device)
        start_epoch = 0
        training_log = {
            'epochs': [], 'train_loss': [], 'train_acc': [],
            'best_f1': 0.0, 'best_mcc': 0.0,
            'hyperparameters': vars(args)
        }
    
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    max_tweet_param = 30
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        loss, acc = train(epoch, trainer, data_paths, adj, max_tweet_param, args.device)
        
        training_log['epochs'].append(epoch)
        training_log['train_loss'].append(loss)
        training_log['train_acc'].append(acc)
        
        if (epoch + 1) % 2000 == 0:
            f1_score, mcc = test_dict(model, adj, data_paths, max_tweet_param)
            
            if f1_score > training_log['best_f1']:
                training_log['best_f1'] = f1_score
                training_log['best_mcc'] = mcc
                print(f"New best model! F1={f1_score:.4f}, MCC={mcc:.4f}")
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'scheduler_state_dict': trainer.scheduler.state_dict(),
                    'criterion_state_dict': trainer.criterion.state_dict(),
                    'f1_score': f1_score,
                    'mcc': mcc,
                    'args': vars(args)
                }
                torch.save(checkpoint,
                          f'model_checkpoints/model_{run_timestamp}_epoch_{epoch}_f1_{f1_score:.4f}.pt')
            
            with open(f'training_logs/training_log_{run_timestamp}.json', 'w') as f:
                json.dump(training_log, f, indent=4)
    
    print("Optimization Finished!")
    final_f1, final_mcc = test_dict(model, adj, data_paths, max_tweet_param)
    
    # Save final model
    final_checkpoint = {
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'scheduler_state_dict': trainer.scheduler.state_dict(),
        'criterion_state_dict': trainer.criterion.state_dict(),
        'final_f1': final_f1,
        'final_mcc': final_mcc,
        'args': vars(args)
    }
    torch.save(final_checkpoint, f'model_checkpoints/model_{run_timestamp}_final.pt')
    
    with open(f'training_logs/training_log_{run_timestamp}.json', 'w') as f:
        json.dump(training_log, f, indent=4)

if __name__ == "__main__":
    main()
