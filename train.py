#!/usr/bin/env python3
import time
import torch
from torchvision import transforms, datasets
from torch.utils.data import SubsetRandomSampler,random_split, DataLoader
import pdb
import argparse
from carbontracker.tracker import CarbonTracker
from sklearn.metrics import precision_recall_curve, average_precision_score
from torchvision.utils import save_image
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
import sys
import numpy as np
import os
from utils.tools import makeLogFile,writeLog,dice,dice_loss,binary_accuracy
from data.datasets import RSNAdataset, LIDCdataset
import sys
import timm
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import bitsandbytes as bnb
from torch.cuda.amp import autocast

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Globally load device identifier
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def evaluate(loader):
    ### Evaluation function for validation/testing
    vl_acc = torch.Tensor([0.]).to(device)
    vl_loss = 0.
    labelsNp = [] 
    predsNp = [] 
    model.eval()

    for i, (inputs, labels) in enumerate(loader):
        b = inputs.shape[0]
        inputs = inputs.to(device)
        labels = labels.to(device)
        if args.half:
            inputs = inputs.half()
            labels = labels.half()
        if args.amp:
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                scores = model(inputs)
        else: 
            # Inference
            scores = model(inputs)
        scores = scores.view(labels.shape).type_as(labels)

        preds = torch.sigmoid(scores.clone())
        loss = loss_fun(scores, labels) 
        vl_loss += loss
        vl_acc += accuracy(labels,preds)

    # Compute AUC over the full (valid/test) set
    vl_acc = vl_acc.item()/len(loader)
    vl_loss = vl_loss.item()/len(loader)
    
    return vl_acc, vl_loss


#### MAIN STARTS HERE ####

parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=20, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--nhid', type=int, default=4, help='Number of hidden features')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--l2', type=float, default=0, help='L2 regularisation')
parser.add_argument('--data', type=str, default='BRATS/LIDC',help='Path to data.')
parser.add_argument('--dataset', type=str, default='BRATS/LIDC',help='Path to data.')
parser.add_argument('--seed', type=int, default=1, help='Random seed')
parser.add_argument('--densenet', action='store_true', 
					default=False, help='Use densenet baseline')
parser.add_argument('--mlp', action='store_true', 
					default=False, help='Use MLP baseline')
parser.add_argument('--cnn', action='store_true', 
					default=False, help='Use CNN baseline')
parser.add_argument('--timm', action='store_true', 
					default=False, help='Use models from timm')
parser.add_argument('--bnb', action='store_true', 
					default=False, help='Use BitsAndBytes')
parser.add_argument('--model_name', type=str, default='densenet121',help='Path to data.')
parser.add_argument('--amp', action='store_true', 
					default=False, help='Use Automatic mixed precision')
parser.add_argument('--downsample', type=int, default=1, help='Downsampling factor')
parser.add_argument('--half', action='store_true', 
					default=False, help='Use half precision model')
parser.add_argument('--inference', type=int, default=0, help='Only perform inference')
parser.add_argument('--lidc', action='store_true', 
					default=False, help='Use LIDC dataset')



args = parser.parse_args()

# Assign script args to vars
torch.manual_seed(args.seed)
batch_size = args.batch_size


### Data processing and loading....
if args.lidc:
    print("Using LIDC dataset")
    dataset = LIDCdataset()
else:
    print("Using RSNA dataset")
    dataset = RSNAdataset(downsample=args.downsample)

x = dataset[0][0]
dim = x.shape[-1]
print('Using %d size of images'%dim)
N = len(dataset)
train_sampler = SubsetRandomSampler(np.arange(int(0.6*N)))
valid_sampler = SubsetRandomSampler(np.arange(int(0.6*N),int(0.8*N)))
test_sampler = SubsetRandomSampler(np.arange(int(0.8*N),N))

# Initialize loss and metrics
loss_fun = torch.nn.BCEWithLogitsLoss()
accuracy = binary_accuracy

# Initiliaze input dimensions
num_train = len(train_sampler)
num_valid = len(valid_sampler)
num_test = len(test_sampler)
print("Num. train = %d, Num. val = %d, Num. test = %d"%(num_train,num_valid,num_test))

# Initialize dataloaders
loader_train = DataLoader(dataset = dataset, drop_last=False,num_workers=0,
        batch_size=batch_size, pin_memory=True,sampler=train_sampler)
loader_valid = DataLoader(dataset = dataset, drop_last=True,num_workers=0,
        batch_size=batch_size, pin_memory=True,sampler=valid_sampler)
loader_test = DataLoader(dataset = dataset, drop_last=True,num_workers=0,
        batch_size=batch_size, pin_memory=True,sampler=test_sampler)

nValid = len(loader_valid)
nTrain = len(loader_train)
nTest = len(loader_test)

if args.densenet:
    print("Densenet Baseline!")
    model = DenseNet(depth=40, growthRate=12,reduction=0.5,bottleneck=True,nClasses=1)
    mName = 'dense'
elif args.mlp:
    print("MLP Baseline!")
    model = BaselineMLP(inCh=dim**2,nhid=args.nhid,nClasses=1)
    mName = 'mlp'
elif args.cnn:
    print("CNN Baseline!")
    model = BaselineCNN(inCh=1,nhid=32,nClasses=1)
    mName = 'cnn'
elif args.timm:
    mName = args.model_name 
    print("Using "+args.model_name+" from TIMM")
    model = timm.create_model(mName, pretrained=True, in_chans=1,num_classes=1)
else:
    print("Aborting execution. Specify model!")
    sys.exit()


model = model.to(device)
if args.half:
    model = model.half()
# Initialize optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, 
                             weight_decay=args.l2)
if args.bnb:
    print("Using BNB for 8bit optimizers")
    optimizer = bnb.optim.Adam8bit(model.parameters(), lr=args.lr, 
                                 weight_decay=args.l2)

if args.amp:
    use_amp = True
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

if args.inference > 0:
    t = time.time()
    with torch.no_grad():
        for i in range(args.inference):
            vl_acc, vl_loss = evaluate(loader_test)
    print('Time:%.2f'%((time.time()-t)/args.inference))
    sys.exit() 


nParam = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Number of parameters:%d"%(nParam))
print(f"Using Adam w/ learning rate = {args.lr:.1e}")

# Miscellaneous initialization
start_time = time.time()
maxAuc = -1
minLoss = 1e3
convCheck = args.num_epochs//10 if args.num_epochs > 10 else 1
convIter = 0

# Visualization and log dirs
logLoc = 'logs/'+time.strftime("%Y%m%d_%H_%M")+'_'+mName+'_D_'+repr(args.downsample)+'_S_%03d'%args.seed
if args.bnb:
    logLoc = logLoc+'_bnb'
if args.amp:
    logLoc = logLoc+'_amp'
if args.half:
    logLoc = logLoc+'_half'


print("Saving in "+logLoc)
if not os.path.exists(logLoc):
    os.mkdir(logLoc)

logFile = logLoc+'/log.txt'
makeLogFile(logFile)

with open(logFile,"a") as f:
    print("Number of parameters:%d"%(nParam))


# Instantiate Carbontracker
tracker = CarbonTracker(epochs=args.num_epochs,
            log_dir=logLoc,monitor_epochs=-1)

# Training starts here
for epoch in range(args.num_epochs):
    tracker.epoch_start()
    running_loss = 0.
    running_acc = 0.
    t = time.time()
    model.train()
    predsNp = [] 
    labelsNp = []
    bNum = 0
    for i, (inputs, labels) in enumerate(loader_train):


        inputs = inputs.to(device)
        labels = labels.to(device)
        if args.half:
            inputs = inputs.half()
            labels = labels.half()

        for p in model.parameters():
            p.grad = None
        bNum += 1
        b = inputs.shape[0]
        # Make patches on the fly
        if args.amp:
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                scores = model(inputs)
                scores = scores.view(labels.shape).type_as(labels)
                loss = loss_fun(scores, labels) 
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:

            scores = model(inputs)
            scores = scores.view(labels.shape).type_as(labels)
            loss = loss_fun(scores, labels) 

        # Backpropagate and update parameters
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            preds = torch.sigmoid(scores.clone())
            running_acc += (accuracy(labels,preds)).item()
            running_loss += loss.item()
            
        if (i+1) % 10 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, args.num_epochs, i+1, nTrain, loss.item()))
    
    tr_acc = running_acc/nTrain

    if epoch == 0:
        t = torch.cuda.get_device_properties(0).total_memory/1e9
        r = torch.cuda.memory_reserved(0)/1e9
        a = torch.cuda.memory_allocated(0)/1e9
        m = torch.cuda.max_memory_allocated(0)/1e9

    # Evaluate on Validation set 
    with torch.no_grad():

        vl_acc, vl_loss = evaluate(loader_valid)
        if vl_acc > maxAuc or vl_loss < minLoss:
            convIter = 0
            if (vl_acc > maxAuc) or (vl_acc >= maxAuc and vl_loss < minLoss):
                ### Predict on test set if new optimum
                maxAuc = vl_acc
                print('New Best: %.4f'%(maxAuc))
                convEpoch = epoch
                ts_acc = ts_loss = 0
                if epoch % 2 == 0:
                    best_ts_acc, best_ts_loss = evaluate(loader=loader_test)
                    print('Test Set Loss:%.4f\t Acc:%.4f'%(best_ts_loss, best_ts_acc))
                    with open(logFile,"a") as f:
                        print('Test Set Loss:%.4f\tAcc:%.4f'%(best_ts_loss, best_ts_acc),file=f)


            elif vl_loss < minLoss:
                minLoss = vl_loss
        else:
            convIter += 1
        if convIter == convCheck:
            print("Converged at epoch:%d with AUC:%.4f"%(convEpoch+1,maxAuc))
            break
    writeLog(logFile, epoch, running_loss/bNum, tr_acc,
            vl_loss, vl_acc, ts_loss, ts_acc,  time.time()-t)
    tracker.epoch_end()
tracker.stop()
print(t,r,a,m)
print('Param: %.4f M, GPU memory:%.4f G '%(nParam/1e6,r+a))
with open(logFile,"a") as f:
        print('Test Acc: %.4f, Test Loss:%.4f'%(best_ts_acc, best_ts_loss),file=f)
        print('Param: %.4f M, GPU memory:%.4f G'%(nParam/1e6,r+a),file=f)
