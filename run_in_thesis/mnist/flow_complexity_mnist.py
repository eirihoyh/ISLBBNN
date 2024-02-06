import math
import numpy as np
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

# define the data loaders
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        './mnist', train=True, download=False,
        transform=transforms.ToTensor()),
    batch_size=BATCH_SIZE, shuffle=True, **LOADER_KWARGS)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        './mnist', train=False, download=False,
        transform=transforms.ToTensor()),
    batch_size=TEST_BATCH_SIZE, shuffle=False, **LOADER_KWARGS)


TRAIN_SIZE = len(train_loader.dataset)
TEST_SIZE = len(test_loader.dataset)
NUM_BATCHES = len(train_loader)
NUM_TEST_BATCHES = len(test_loader)

assert (TRAIN_SIZE % BATCH_SIZE) == 0
assert (TEST_SIZE % TEST_BATCH_SIZE) == 0

p = 28*28



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


import torchvision
trainset = torchvision.datasets.MNIST(root='./mnist', train=True, download=False, transform=transforms.ToTensor())
testset = torchvision.datasets.MNIST(root='./mnist', train=False, download=False, transform=transforms.ToTensor())

train_data = trainset.data/255.
train_target = trainset.targets

test_data = testset.data/255.
test_target = testset.targets


all_data = torch.cat((train_data, test_data), 0)
all_targets = torch.cat((train_target, test_target), 0)
random_numbers = np.random.randint(0,70000,size=5000)
rand_data = all_data[random_numbers,:,:]

del all_targets
del test_data
del test_target
del trainset
del testset
del all_data
del random_numbers

second_der_mc = {}
complexity_class = {}
complexity_tot = {}
second_ders = {}
for i in range(10):
    net = torch.load(f"network/flow_class/net{i}", map_location=torch.device('cpu'))
    mc, ccc, cc, sder = pip_func.complexity_measure_mnist(net,p,rand_data)
    second_der_mc[i] = mc
    complexity_class[i] = ccc
    complexity_tot[i] = cc
    second_ders[i] = sder
    print(f"Done net {i}")

np.save("complexity/flow_class/second_der_mc", second_der_mc)
np.save("complexity/flow_class/complexity_class", complexity_class)
np.save("complexity/flow_class/complexity_tot", complexity_tot)
np.save("complexity/flow_class/second_ders", second_ders)



