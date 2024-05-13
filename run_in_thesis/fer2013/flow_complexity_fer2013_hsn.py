import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm, trange
import pipeline_functions as pip_func
import os
import sys
import importlib
current_dir = os.getcwd()
sys.path.append('layers')
from config import config
from flow_layers import BayesianLinear
from torch.optim.lr_scheduler import MultiStepLR

# select the device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LOADER_KWARGS = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
# cuda = torch.cuda.set_device(0)

if (torch.cuda.is_available()):
    print("GPUs are used!")
else:
    print("CPUs are used!")
    
    
# define the parameters
BATCH_SIZE = 100
TEST_BATCH_SIZE = 1000
CLASSES = 10
SAMPLES = 1
TEST_SAMPLES = 10

# define parameters
HIDDEN_LAYERS = config['n_layers'] - 2 
epochs = config['num_epochs']
dim = config['hidden_dim']
num_transforms = config['num_transforms']
n_nets = config['n_nets']
lr = config['lr']
verbose = config['verbose']
save_res = config['save_res']
# patience = config['patience']

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
        self.linears = nn.ModuleList([BayesianLinear(p, dim, a_prior=0.1, num_transforms=num_transforms)])
        self.linears.extend([BayesianLinear((dim+p), (dim), a_prior=0.1, num_transforms=num_transforms) for _ in range(HIDDEN_LAYERS-1)])
        self.linears.append(BayesianLinear((dim+p), CLASSES, a_prior=0.1, num_transforms=num_transforms))
        self.loss = nn.BCELoss(reduction='sum')  # Setup loss (Binary cross entropy as binary classification)
        

    def forward(self, x, sample=False, ensemble=False):
        x_input = x.view(-1, p)
        x = F.sigmoid(self.linears[0](x_input, ensemble))
        i = 1
        for l in self.linears[1:-1]:
            x = F.sigmoid(l(torch.cat((x, x_input),1), ensemble))
            i += 1

        out = F.log_softmax((self.linears[i](torch.cat((x, x_input),1), ensemble)), dim=1)
        return out
    
    def forward_preact(self, x, sample=False, ensemble=False, calculate_log_probs=False):
        x_input = x.view(-1, p)
        x = F.sigmoid(self.linears[0](x_input, ensemble))
        i = 1
        for l in self.linears[1:-1]:
            x = F.sigmoid(l(torch.cat((x, x_input),1), ensemble))
            i += 1

        out = self.linears[i](torch.cat((x, x_input),1), ensemble)
        return out

    def kl(self):
        kl_sum = self.linears[0].kl_div()
        for l in self.linears[1:]:
            kl_sum = kl_sum + l.kl_div()
        return kl_sum 


all_data = torch.tensor(X,dtype=torch.float32)
print(all_data.shape)
all_targets = torch.tensor(y)
random_numbers = np.random.randint(0,n,size=10)
rand_data = all_data[random_numbers]
print(rand_data.shape)
print(rand_data[0])
print(rand_data[0,:])
del all_targets
del all_data
del random_numbers

second_der_mc = {}
complexity_class = {}
complexity_tot = {}
second_ders = {}
for i in range(10):
    net = torch.load(f"network/flow_class/net{i}_hsn_0.0025lr_sigmoid", map_location=torch.device('cpu'))
    net.eval()
    print(net.forward_preact(rand_data[0]))
    mc, ccc, cc, sder = pip_func.complexity_measure_fer2013(net,p,rand_data)
    second_der_mc[i] = mc
    complexity_class[i] = ccc
    complexity_tot[i] = cc
    second_ders[i] = sder
    print(f"Done net {i}")

np.save("complexity/flow_class/second_der_mc", second_der_mc)
np.save("complexity/flow_class/complexity_class", complexity_class)
np.save("complexity/flow_class/complexity_tot", complexity_tot)
np.save("complexity/flow_class/second_ders", second_ders)



