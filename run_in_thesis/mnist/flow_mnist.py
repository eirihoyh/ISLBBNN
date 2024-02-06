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
BATCH_SIZE = 3000
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
        './mnist', train=True, download=True,
        transform=transforms.ToTensor()),
    batch_size=BATCH_SIZE, shuffle=True, **LOADER_KWARGS)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        './mnist', train=False, download=True,
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

    def kl(self):
        kl_sum = self.linears[0].kl_div()
        for l in self.linears[1:]:
            kl_sum = kl_sum + l.kl_div()
        return kl_sum 
    

# Stochastic Variational Inference iteration
def train(net, optimizer, epoch):
    net.train()
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(DEVICE), target.to(DEVICE)
        net.zero_grad()
        outputs = net(data, ensemble=True)
        # negative_log_likelihood = net.loss(outputs, target)
        negative_log_likelihood = F.nll_loss(outputs, target, reduction="sum")
        #if epoch <= 1000:
        #    loss = negative_log_likelihood + (epoch/1000)*(net.kl() / NUM_BATCHES)
        #else:
        loss = negative_log_likelihood + net.kl() / NUM_BATCHES
        loss.backward()
        optimizer.step()
    if verbose:
        alpha_clean = pip_func.clean_alpha(net, threshold=0.5)
        density_median, used_weigths_median, _ = pip_func.network_density_reduction(alpha_clean)
        print('loss', loss.item())
        print('nll', negative_log_likelihood.item())
        print('density', pip_func.expected_number_of_weights(net, p)/nr_weights)
        print("median weights", used_weigths_median)
        print('')
    return negative_log_likelihood.item(), loss.item()


def test_ensemble(net):
    net.eval()
    metr = []
    ensemble = []
    median = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            outputs = torch.zeros(TEST_SAMPLES, TEST_BATCH_SIZE, CLASSES).to(DEVICE)
            out2 = torch.zeros_like(outputs)
            for i in range(TEST_SAMPLES):
                outputs[i] = net.forward(data, sample=True, ensemble=True)  # model avg over structures and weights
                out2[i] = net.forward(data, sample=True, ensemble=False)  # only model avg over weights where a > 0.5

            output1 = outputs.mean(0)
            out2 = out2.mean(0)

            pred1 = output1.max(1, keepdim=True)[1]  # index of max log-probability
            pred2 = out2.max(1, keepdim=True)[1]

            a = pred2.eq(target.view_as(pred2)).sum().item()
            b = pred1.eq(target.view_as(pred1)).sum().item()
            median.append(a)
            ensemble.append(b)
    # estimate hte sparsity
    alpha_clean = pip_func.clean_alpha(net, threshold=0.5)
    density_median, used_weigths_median, _ = pip_func.network_density_reduction(alpha_clean)
    # density.append(density_median)
    # used_weights.append(used_weigths_median)
    # g1 = ((net.l1.alpha_q.detach().cpu().numpy() > 0.5) * 1.)
    # g2 = ((net.l2.alpha_q.detach().cpu().numpy() > 0.5) * 1.)
    # g3 = ((net.l3.alpha_q.detach().cpu().numpy() > 0.5) * 1.)
    # gs = np.concatenate((g1.flatten(), g2.flatten(), g3.flatten()))
    metr.append(np.sum(median) / TEST_SIZE)
    metr.append(np.sum(ensemble) / TEST_SIZE)
    metr.append(density_median.cpu().detach().numpy())
    metr.append(used_weigths_median.cpu().detach().numpy())
    if verbose:
        print(density_median, 'sparsity')
        print(used_weigths_median, 'nr weights')
        print(np.sum(median) / TEST_SIZE, 'median')
        print(np.sum(ensemble) / TEST_SIZE, 'ensemble')
    return metr


import time

BATCH_SIZE_LIST = [2500, 3000, 4000]
for BATCH_SIZE in BATCH_SIZE_LIST:
    nll_several_runs = []
    loss_several_runs = []
    metrics_several_runs = []
    
    # make inference on 10 networks
    for i in range(n_nets):
        print('network', i)
        torch.manual_seed(i)
        net = BayesianNetwork().to(DEVICE)
        alphas = pip_func.get_alphas_numpy(net)
        nr_weights = np.sum([np.prod(a.shape) for a in alphas])
        print(f'Tot weights in model: {nr_weights}')
        optimizer = optim.Adam(net.parameters(), lr=lr)
        all_nll = []
        all_loss = []
        t1 = time.time()
        for epoch in range(epochs):
            print('epoch', epoch)
            nll, loss = train(net, optimizer,epoch)
            all_nll.append(nll)
            all_loss.append(loss)
            
                
        nll_several_runs.append(all_nll)
        loss_several_runs.append(all_loss)
        t = round((time.time() - t1), 1)
        if save_res:
            torch.save(net, f"network/flow_class/net{i}_large_{BATCH_SIZE}_bs")
        metrics = test_ensemble(net)
        metrics.append(t / epochs)
        metrics_several_runs.append(metrics)
        
    
    if save_res:
        np.savetxt(f'results/flow_class/MNIST_KL_loss_FLOW_{HIDDEN_LAYERS}_hidden_{dim}_dim_{lr}_lr_{num_transforms}_num_trans_LARGE_{BATCH_SIZE}_bs' + '.txt', loss_several_runs, delimiter=',')
        np.savetxt(f'results/flow_class/MNIST_KL_metrics_FLOW_{HIDDEN_LAYERS}_hidden_{dim}_dim_{lr}_lr_{num_transforms}_num_trans_LARGE_{BATCH_SIZE}_bs' + '.txt', metrics_several_runs, delimiter=',')
        np.savetxt(f'results/flow_class/MNIST_KL_nll_FLOW_{HIDDEN_LAYERS}_hidden_{dim}_dim_{lr}_lr_{num_transforms}_num_trans_LARGE_{BATCH_SIZE}_bs' + '.txt', nll_several_runs, delimiter=',')