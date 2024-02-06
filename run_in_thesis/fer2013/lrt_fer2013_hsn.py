#----------Imports----------
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
from scipy import stats
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold, train_test_split
import pipeline_functions as pip_func
# import importlib
from torchmetrics import R2Score
import os
import sys
current_dir = os.getcwd()
sys.path.append('layers')
from config import config
from lrt_layers import BayesianLinear


# select the device
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "mps")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LOADER_KWARGS = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}



# define parameters
HIDDEN_LAYERS = config['n_layers'] - 2 
epochs = config['num_epochs']
dim = config['hidden_dim']
num_transforms = config['num_transforms']
n_nets = config['n_nets']
lr = config['lr']
verbose = config['verbose']
save_res = config['save_res']
patience = config['patience']
SAMPLES = 1

df = pd.read_csv("data/icml_face_data.csv")

data = np.zeros((len(df), 48*48))
for i in range(len(data)):
    data[i] = np.array(df[" pixels"][i].split(" ")).astype(np.float32)/255.
targets = df.emotion.values

X,y  = data, targets
used_inds = [yi in [3,5,6] for yi in y]
y = y[used_inds]
y = np.where(y == 3, 0, y)
y = np.where(y == 5, 1, y)
y = np.where(y == 6, 2, y)
X = X[used_inds]

n,p = X.shape
print(n,p,dim)

CLASSES = len(np.unique(y))


#-------SKIP CONNECTION LBBNN--------

class BayesianNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # set the architecture
        self.linears = nn.ModuleList([BayesianLinear(p, dim, a_prior=0.05)])
        self.linears.extend([BayesianLinear((dim+p), (dim), a_prior=0.05) for _ in range(HIDDEN_LAYERS-1)])
        self.linears.append(BayesianLinear((dim+p), CLASSES, a_prior=0.05))
        self.loss = nn.BCELoss(reduction='sum')  # Setup loss (Binary cross entropy as binary classification)
        #self.loss = nn.MSELoss(reduction='sum')

    def forward(self, x, sample=False, ensemble=False):
        x_input = x.view(-1, p)
        x = F.sigmoid(self.linears[0](x_input, ensemble, sample))
        i = 1
        for l in self.linears[1:-1]:
            x = F.sigmoid(l(torch.cat((x, x_input),1), ensemble, sample))
            i += 1

        out = F.log_softmax(self.linears[i](torch.cat((x, x_input),1), ensemble, sample))
        #out = self.linears[i](torch.cat((x, x_input),1), ensemble, sample)
        return out

    def kl(self):
        kl_sum = self.linears[0].kl
        for l in self.linears[1:]:
            kl_sum = kl_sum + l.kl
        return kl_sum 


# Define BATCH sizes
BATCH_SIZE = int((n*0.8)/10)
TEST_BATCH_SIZE = int(n*0.10) # W|ould normally call this the "validation" part (will be used during training)
VAL_BATCH_SIZE = int(n*0.10) # and this the "test" part (will be used after training)

TRAIN_SIZE = int((n*0.80)/10)
TEST_SIZE = int(n*0.10) # Would normally call this the "validation" part (will be used during training)
VAL_SIZE = int(n*0.10) # and this the "test" part (will be used after training)

NUM_BATCHES = TRAIN_SIZE/BATCH_SIZE

# Split keep some of the data for validation after training
X, X_test, y, y_test = train_test_split(
    X, y, test_size=0.10, random_state=42, stratify=y)

#------TRAIN AND TEST-------

nll_several_runs = []
loss_several_runs = []
metrics_several_runs = []
metrics_median_several_runs = []

all_nets = {}
for ni in range(n_nets):
    print('network', ni)
    torch.manual_seed(ni)
    net = BayesianNetwork().to(DEVICE)
    alphas = pip_func.get_alphas_numpy(net)
    nr_weights = np.sum([np.prod(a.shape) for a in alphas])
    print(f'Tot weights in model: {nr_weights}')

    optimizer = optim.Adam(net.parameters(), lr=lr)
    
    all_nll = []
    all_loss = []

    # Split into training and test set
    X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=1/9, random_state=ni, stratify=y)

    test_dat = torch.tensor(np.column_stack((X_test,y_test)),dtype = torch.float32)
            
    train_dat = torch.tensor(np.column_stack((X_train,y_train)),dtype = torch.float32)
    val_dat = torch.tensor(np.column_stack((X_val,y_val)),dtype = torch.float32)
    
    # Train network
    counter = 0
    highest_acc = 0
    best_model = copy.deepcopy(net)
    for epoch in range(epochs):
        if verbose:
            print(epoch)
        nll, loss = pip_func.train(net, train_dat, optimizer, BATCH_SIZE, NUM_BATCHES, p, DEVICE, nr_weights, flows=True, multiclass=True)
        nll_val, loss_val, ensemble_val = pip_func.val(net, val_dat, DEVICE, flows=True, verbose=verbose, reg=False, multiclass=True)
        if ensemble_val >= highest_acc and epoch>3000:
            counter = 0
            highest_acc = ensemble_val
            best_model = copy.deepcopy(net)
        else:
            counter += 1
        
        all_nll.append(nll)
        all_loss.append(loss)

        if counter >= patience:
            break
    

        
    # EVAL part
    nll_several_runs.append(all_nll)
    loss_several_runs.append(all_loss)
    all_nets[ni] = best_model 
    # all_nets[ni] = net
    # Save all nets for later
    if save_res:
        torch.save(all_nets[ni], f"network/lrt_class/net{ni}_hsn")
    metrics, metrics_median = pip_func.test_ensemble(all_nets[ni],test_dat,DEVICE, SAMPLES=10, flows=True, reg=False, verbose=verbose, multiclass=True) # Test same data 10 times to get average 
    metrics_several_runs.append(metrics)
    metrics_median_several_runs.append(metrics_median)

if verbose:
    print(metrics)
if save_res:
    m = np.array(metrics_several_runs)
    np.savetxt(f'results/lrt_class_skip_{HIDDEN_LAYERS}_hidden_{dim}_dim_{epochs}_epochs_{lr}_lr_fer2013_hsn_full.txt',m,delimiter = ',')
    m_median = np.array(metrics_median_several_runs)
    np.savetxt(f'results/lrt_class_skip_{HIDDEN_LAYERS}_hidden_{dim}_dim_{epochs}_epochs_{lr}_lr_fer2013_hsn_median.txt',m_median,delimiter = ',')