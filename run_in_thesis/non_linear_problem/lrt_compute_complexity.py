#----------Imports----------
import math
import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy import stats
import string as s
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
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LOADER_KWARGS = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

print(DEVICE)



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

y, X = pip_func.create_data_unif(classification=True)

n, p = X.shape  # need this to get p 
print(n,p,dim)


#-------SKIP CONNECTION LBBNN--------

class BayesianNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # set the architecture
        self.linears = nn.ModuleList([BayesianLinear(p, dim, a_prior=0.1)])
        self.linears.extend([BayesianLinear((dim+p), (dim), a_prior=0.1) for _ in range(HIDDEN_LAYERS-1)])
        self.linears.append(BayesianLinear((dim+p), 1, a_prior=0.1))
        self.loss = nn.BCELoss(reduction='sum')  # Setup loss (Binary cross entropy as binary classification)
        

    def forward(self, x, sample=False, ensemble=False):
        x_input = x.view(-1, p)
        x = F.sigmoid(self.linears[0](x_input, ensemble, sample))
        i = 1
        for l in self.linears[1:-1]:
            x = F.sigmoid(l(torch.cat((x, x_input),1), ensemble, sample))
            i += 1

        out = torch.sigmoid(self.linears[i](torch.cat((x, x_input),1), ensemble, sample))
        return out
    
    def forward_preact(self, x, sample=False, ensemble=False, calculate_log_probs=False):
        x_input = x.view(-1, p)
        x = F.sigmoid(self.linears[0](x_input, ensemble, sample))
        i = 1
        for l in self.linears[1:-1]:
            x = F.sigmoid(l(torch.cat((x, x_input),1), ensemble, sample))
            i += 1

        out = (self.linears[i](torch.cat((x, x_input),1), ensemble, sample))
        return out

    def kl(self):
        kl_sum = self.linears[0].kl_div
        for l in self.linears[1:]:
            kl_sum = kl_sum + l.kl_div
        return kl_sum 

# #------ y = x_1 + x_2 + 100, with x_3 dep on x_1----------
# first_der = {}
# second_der = {}
# complexity = {}
# dep_levels = [0,10,50,90]
# for d in dep_levels:
#     first_der[f"dep: {d}"] = {}
#     second_der[f"dep: {d}"] = {}
#     complexity[f"dep: {d}"] = {}
#     for i in range(10):
#         net = torch.load(f"network/lrt_class/net{i}_unif_{d}", map_location=torch.device('cpu'))
#         rand_numbs = torch.FloatTensor(100000, 4).uniform_(-10,10)
#         first, second, combined_complexity = pip_func.complexity_measure(net, p, rand_numbs)
#         first_der[f"dep: {d}"][i] = first
#         second_der[f"dep: {d}"][i] = second
#         complexity[f"dep: {d}"][i] = combined_complexity

# np.save("complexity/lrt_class/complexity", complexity)
# np.save("complexity/lrt_class/second_der", second_der)
# np.save("complexity/lrt_class/first_der", first_der)


#-------- y = x_1 + x_2 + x_1 x_2 + x_1^2 + x_2^2 + 100, with x_3 dep on x_1 -----------------
first_der = {}
second_der = {}
complexity = {}
dep_levels = [0,10,50,90]
for d in dep_levels:
    first_der[f"dep: {d}"] = {}
    second_der[f"dep: {d}"] = {}
    complexity[f"dep: {d}"] = {}
    for i in range(10):
        net = torch.load(f"network/lrt_class/net{i}_unif_dep_{d}", map_location=torch.device('cpu'))
        rand_numbs = torch.FloatTensor(100000, 4).uniform_(-10,10)
        first, second, combined_complexity = pip_func.complexity_measure(net, p, rand_numbs)
        first_der[f"dep: {d}"][i] = first
        second_der[f"dep: {d}"][i] = second
        complexity[f"dep: {d}"][i] = combined_complexity


np.save("complexity/lrt_class/complexity", complexity)
np.save("complexity/lrt_class/second_der", second_der)
np.save("complexity/lrt_class/first_der", first_der)


