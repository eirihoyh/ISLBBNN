import copy
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from config import config
import plot_functions as pf
import pipeline_functions as pip_func
import sys
sys.path.append('networks')
from lrt_net import BayesianNetwork

# define parameters
HIDDEN_LAYERS = config['n_layers'] - 2 
epochs = config['num_epochs']
post_train_epochs = config['post_train_epochs']
dim = config['hidden_dim']
num_transforms = config['num_transforms']
n_nets = config['n_nets']
lr = config['lr']
class_problem = config["class_problem"]
verbose = config['verbose']
save_res = config['save_res']
patience = config['patience']
SAMPLES = 1




#---------DATA------------
df = pd.read_csv("abalone.csv")
df = pd.get_dummies(df, drop_first=True)*1
#df = df.drop(["Sex"])
X_original = df.loc[:,df.columns != "Rings"].values
y_original = df.loc[:,df.columns == "Rings"].values.T[0]
n, p = X_original.shape  # need this to get p 

print(n,p)


all_data = torch.tensor(X_original,dtype = torch.float32)
draw = 1_000

first_der_mc = {}
second_der_mc = {}
complexity_tot = {}
for i in range(10):
    net = torch.load(f"trained_nets/lrt/net{i}", map_location=torch.device('cpu'))
    rand_ints = np.random.choice(n,draw)
    rand_numbs = copy.deepcopy(all_data)[rand_ints]
    mc_first, mc_int_second, cc =pip_func.complexity_measure(net, p, rand_numbs)
    first_der_mc[i] = mc_first
    second_der_mc[i] = mc_int_second
    complexity_tot[i] = cc
    print(f"Done net {i}")

np.save("complexity/lrt/second_der_mc", second_der_mc)
np.save("complexity/lrt/complexity_tot", complexity_tot)
np.save("complexity/lrt/first_der_mc", first_der_mc)


print(complexity_tot)


print(second_der_mc)