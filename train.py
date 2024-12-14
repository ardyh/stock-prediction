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
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef, confusion_matrix
from sklearn.metrics import classification_report
import pickle
import json
from datetime import datetime

from utils import load_data, accuracy
from model import GAT #, SpGAT

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--seed', type=int, default=14, help='Random seed.')
parser.add_argument('--epochs', type=int, default=5000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.38, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
parser.add_argument('--continue_epochs', type=int, default=None, help='Number of epochs to continue training')

args = parser.parse_args()
# args.cuda = not args.no_cuda and torch.cuda.is_available()
args.cuda=True
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj = load_data()
stock_num = adj.size(0)

train_price_path = "train_price/"
train_label_path = "train_label/"
train_text_path = "train_text/"
val_price_path = "val_price/"
val_label_path = "val_label/"
val_text_path = "val_text/"
test_price_path = "test_price/"
test_label_path = "test_label/"
test_text_path = "test_text/"
num_samples = len(os.listdir(train_price_path))
import os
import time
import pickle
import datetime
import numpy as np
from sklearn.metrics import classification_report, matthews_corrcoef, accuracy_score
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt

cross_entropy = nn.CrossEntropyLoss(weight=torch.tensor([1.00,1.00]).cuda())

# Create directories if they don't exist
os.makedirs('model_checkpoints', exist_ok=True)
os.makedirs('training_logs', exist_ok=True)

# Create a unique identifier for this training run
run_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

# Initialize a dictionary to store training metrics
training_log = {
    'epochs': [],
    'train_loss': [],
    'train_acc': [],
    'best_f1': 0.0,
    'best_mcc': 0.0,
    'hyperparameters': vars(args)  # Save all hyperparameters
}

max_tweet_param = 10

# 390 batches
def train(epoch): # 777 mod 390 = [0 ]
    t = time.time()
    model.train()
    optimizer.zero_grad()
    i = epoch % 394
    train_text = torch.tensor(np.load(train_text_path+str(i).zfill(10)+'.npy')[:, :, :max_tweet_param, :], dtype=torch.float32).cuda()
    train_price = torch.tensor(np.load(train_price_path+str(i).zfill(10)+'.npy'), dtype = torch.float32).cuda()
    train_label = torch.LongTensor(np.load(train_label_path+str(i).zfill(10)+'.npy')).cuda()
    output = model(train_text, train_price, adj)
    loss_train = cross_entropy(output, torch.max(train_label,1)[1])
    acc_train = accuracy(output, torch.max(train_label,1)[1])
    loss_train.backward()
    optimizer.step()

    print(f"Epoch {epoch}, batch {i}, Loss: {loss_train.item()}, Accuracy: {acc_train.item()}")
    return loss_train.item(), acc_train.item()

def test_dict():
    pred_dict = dict()
    with open('label_data.p', 'rb') as fp:
        true_label = pickle.load(fp)
    with open('price_feature_data.p', 'rb') as fp:
        feature_data = pickle.load(fp)
    with open('text_feature_data.p', 'rb') as fp:
        text_ft_data = pickle.load(fp)
    model.eval()
    test_acc = []
    test_loss = []
    li_pred = []
    li_true = []
    for dates in feature_data.keys():
        test_text = torch.tensor(text_ft_data[dates][:, :, :max_tweet_param, :],dtype=torch.float32).cuda()
        test_price = torch.tensor(feature_data[dates],dtype=torch.float32).cuda()
        test_label = torch.LongTensor(true_label[dates]).cuda()
        output = model(test_text, test_price,adj)
        output = F.softmax(output, dim=1)
        pred_dict[dates] = output.cpu().detach().numpy()
        loss_test = F.nll_loss(output, torch.max(test_label,1)[0])
        acc_test = accuracy(output, torch.max(test_label,1)[1])
        a = torch.max(output,1)[1].cpu().numpy()
        b = torch.max(test_label,1)[1].cpu().numpy() 
        li_pred.append(a)
        li_true.append(b)
        test_loss.append(loss_test.item())
        test_acc.append(acc_test.item())
    iop = f1_score(np.array(li_true).reshape((-1,)),np.array(li_pred).reshape((-1,)), average='micro')
    mat = matthews_corrcoef(np.array(li_true).reshape((-1,)),np.array(li_pred).reshape((-1,)))
    print("Test set results:",
          "loss= {:.4f}".format(np.array(test_loss).mean()),
          "accuracy= {:.4f}".format(np.array(test_acc).mean()),
          "F1 score={:.4f}".format(iop),
          "MCC = {:.4f}".format(mat))
    with open('pred_dict.p', 'wb') as fp:
        pickle.dump(pred_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
    return iop, mat

def load_checkpoint(checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}")
    # Load checkpoint to CPU first to avoid potential GPU memory issues
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Recreate model with same parameters
    model = GAT(nfeat=64,
                nhid=checkpoint['args']['hidden'],
                nclass=2,
                dropout=checkpoint['args']['dropout'],
                nheads=checkpoint['args']['nb_heads'],
                alpha=checkpoint['args']['alpha'],
                stock_num=stock_num)
    
    # Move model to GPU if available
    if args.cuda:
        model = model.cuda()
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(),
                          lr=checkpoint['args']['lr'],
                          weight_decay=checkpoint['args']['weight_decay'])
    
    # Load optimizer state and move to GPU if needed
    if args.cuda:
        for key in checkpoint['optimizer_state_dict']:
            if isinstance(checkpoint['optimizer_state_dict'][key], dict):
                for k, v in checkpoint['optimizer_state_dict'][key].items():
                    if isinstance(v, torch.Tensor):
                        checkpoint['optimizer_state_dict'][key][k] = v.cuda()
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load training log
    training_log = {
        'epochs': [],
        'train_loss': [],
        'train_acc': [],
        'best_f1': checkpoint.get('f1_score', 0.0),
        'best_mcc': checkpoint.get('mcc', 0.0),
        'hyperparameters': checkpoint['args']
    }
    
    return model, optimizer, checkpoint['epoch'], training_log

# Modify the main training section
if args.resume:
    # Load the checkpoint
    model, optimizer, start_epoch, training_log = load_checkpoint(args.resume)
    
    # Move adj to GPU if needed
    if args.cuda:
        adj = adj.cuda()
    
    # If continue_epochs is specified, adjust total epochs
    if args.continue_epochs is not None:
        args.epochs = start_epoch + args.continue_epochs
    
    print(f"Resuming from epoch {start_epoch} to {args.epochs}")
else:
    # Normal initialization
    model = GAT(nfeat=64, 
                nhid=args.hidden, 
                nclass=2, 
                dropout=args.dropout, 
                nheads=args.nb_heads, 
                alpha=args.alpha,
                stock_num=stock_num)
    if args.cuda:
        model.cuda()
        adj = adj.cuda()
    
    optimizer = optim.Adam(model.parameters(), 
                       lr=args.lr, 
                       weight_decay=args.weight_decay)
    
    start_epoch = 0
    training_log = {
        'epochs': [],
        'train_loss': [],
        'train_acc': [],
        'best_f1': 0.0,
        'best_mcc': 0.0,
        'hyperparameters': vars(args)
    }

# Training loop
best_f1 = training_log['best_f1']
for epoch in range(start_epoch, args.epochs):
    # Training
    loss, acc = train(epoch)
    
    # Save training metrics
    training_log['epochs'].append(epoch)
    training_log['train_loss'].append(loss)
    training_log['train_acc'].append(acc)
    
    # Every N epochs, run validation and save if better
    if (epoch + 1) % 2000 == 0:  # Adjust frequency as needed
        f1_score, mcc = test_dict()
        
        # Save best model
        if f1_score > training_log['best_f1']:
            training_log['best_f1'] = f1_score
            training_log['best_mcc'] = mcc

            # print best f1 and mcc
            print(f"Best F1 score: {training_log['best_f1']:.4f}, Best MCC: {training_log['best_mcc']:.4f}")
            
            # # Save model checkpoint
            # checkpoint = {
            #     'epoch': epoch,
            #     'model_state_dict': model.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            #     'f1_score': f1_score,
            #     'mcc': mcc,
            #     'args': vars(args)
            # }
            # torch.save(
            #     checkpoint,
            #     f'model_checkpoints/model_{run_timestamp}_epoch_{epoch}_f1_{f1_score:.4f}.pt'
            # )
        
        # Save current training log
        # with open(f'training_logs/training_log_{run_timestamp}.json', 'w') as f:
        #     json.dump(training_log, f, indent=4)

print("Optimization Finished!")
results = test_dict()

# Save final model and training log
final_checkpoint = {
    'epoch': args.epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'final_f1': results[0],
    'final_mcc': results[1],
    'args': vars(args)
}
torch.save(
    final_checkpoint,
    f'model_checkpoints/model_{run_timestamp}_final.pt'
)

# Save final training log
with open(f'training_logs/training_log_{run_timestamp}.json', 'w') as f:
    json.dump(training_log, f, indent=4)
