import numpy as np
from scipy import stats
import copy

import torch
from torchmetrics import R2Score
import torch.nn.functional as F

import os
import sys
current_dir = os.getcwd()
# sys.path.append('layers')
sys.path.append('layers')
from config import config
dim = config['hidden_dim']
n_layers = config['n_layers'] 
n_hidden_layers = n_layers - 2
n_samples = config['n_samples']



def create_data_unif(n=n_samples,beta=[100,1,1,1,1], dep_level=0.5,classification=False, non_lin=True):
    # Create data
    np.random.seed(47)
    x1 = np.random.uniform(-10,10,n)
    x2 = np.random.uniform(-10,10 ,n)
    x3 = np.random.uniform(-10,10,n)
    x4 = np.random.uniform(-10,10 ,n)

    x3 = dep_level*x1 + (1-dep_level)*x3  # make x3 dependent on x1

    if non_lin:
        y = beta[0] + beta[1]*x1 + beta[2]*x2 + beta[3]*x1**2 + beta[4]*x2**2 + x1*x2 # non-linear model
    else:
        y = beta[0] + beta[1]*x1 + beta[2]*x2
    
    rand0 = stats.norm.rvs(scale=0.01, size=n)
    y += rand0
    if classification:
        y -= y.min()
        y /= y.max()
        # y = np.round(y)
        y = (y > np.median(y))*1

    
    return y, np.concatenate((np.array([x1]).T, np.array([x2]).T, np.array([x3]).T, np.array([x4]).T), axis=1)


def get_alphas(neur_net, n_hidden_layers=n_hidden_layers):
    alphas = {}
    for name, param in neur_net.named_parameters():
        # if param.requires_grad:
        for i in range(n_hidden_layers+1):
            #if f"linears{i}.lambdal" in name:
            if f"linears.{i}.lambdal" in name:
            #if f"l{i+1}.lambdal" in name:
                alphas[i] = copy.deepcopy(1 / (1 + np.exp(-param.cpu().data)))

    return list(alphas.values())


def get_alphas_numpy(net):
    '''
    Get that all alpha matrices are numpy arrays instead
    of tensors
    '''
    a = get_alphas(net)
    for i in range(len(a)):
        a[i] = a[i].detach().numpy()

    return a


def clean_alpha(net, threshold, dim=dim):
    '''
    The alpha list should go from input layer to output layer
    '''
    alpha_list = get_alphas(net)
    clean_dict = {}
    for ind, alpha in enumerate(alpha_list):
        clean_dict[ind] = (alpha > threshold)*1
    for ind in np.arange(1, len(alpha_list))[::-1]:
        clean_dict[ind-1] = (clean_dict[ind-1].T*(sum(clean_dict[ind][:,:dim])>0)).T*1
    for ind in np.arange(1,len(alpha_list)):
        clean_dict[ind] = torch.cat(((clean_dict[ind][:,:dim]*(sum(clean_dict[ind-1].T)>0))*1, clean_dict[ind][:,dim:]), 1)

    return list(clean_dict.values())

def network_density_reduction(clean_alpha_list):
    used_weights = 0
    tot_weights = 0
    for a in clean_alpha_list:
        shape_a = a.shape
        used_weights += sum(sum(a))
        tot_weights += shape_a[0]*shape_a[1]

    return used_weights/tot_weights, used_weights, tot_weights

def create_layer_name_list(n_layers=n_layers):
    layers = ["I"]

    for layer in range(n_layers-2):
        layers.append(f"H{layer+1}")

    layers.append("Output")
    return layers

def input_inclusion_prob(net, p):
    '''
    Gives probability of input going all the way to the output 
    from a given layer.
    THINK that this actually gives the expected number of nodes
    from an input possition (in all layers) to output.  
    This sums all possible paths for the input nodes to go all the
    way to the output. This will happen for all input nodes from each
    layer. 
    '''
    a = get_alphas_numpy(net)
    length = len(a)
    prob_paths = {}
    layer_names = create_layer_name_list()
    for name in layer_names[:-1]:
        for i in range(p):
            prob_paths[f"Prob I{i} from {name}"] = 0

    # Limit how many matrices that will be multiplied
    lims = np.arange(1, length, 1)[::-1]
    for i, name in enumerate(layer_names[:-1]):

        # i == lims[0] means that we are at last matrix
        if i == lims[0]:
            probs = a[i][:,-p:].T
        else:
            count = 0
            probs = a[count+i][:,-p:].T
            while count < lims[i]:
                probs = probs@a[i+count+1][:,:dim].T
                count += 1
        for xi in range(p):
            prob_paths[f"Prob I{xi} from {name}"] = probs[xi][0]
    
    return prob_paths 

def expected_number_of_weights(net, p):
    '''
    Expected number of weights used in the full model. 
    This is simply the sum over all inclusion probabilites.
    Formula: sum over every weight --> sum over all path probs
    '''
    return sum([sum(list(a.flatten())) for a in get_alphas_numpy(net)])

def second_derivative_output(net, p, rand_numbs):
    '''
    Will be used to calcualte our complexity measure.
    The seocnd derivative for each input will give us an idea of how much 
    the "curve" (with respect to an input) changes in our function. We will,
    however, use a Monte Carlo variation in the complexity measure function, so
    we will only calcualte second derivatives to random inputs. 
    '''
    first = {}
    second = {}
    for i in range(p):
        first[i] = []
        second[i] = []

    for rand in rand_numbs:
        net.eval()
        _x = rand.clone().detach().requires_grad_(True)
        out = net.forward_preact(_x, calculate_log_probs=True)
        first_der = torch.autograd.grad(out, _x, create_graph=True, retain_graph=True)[0]
        for i in range(p):
            first[i].append(first_der[i].cpu().detach().numpy())
            second[i].append(torch.autograd.grad(first_der[i], _x, retain_graph=True)[0][i].cpu().detach().numpy())

    return first, second

def complexity_measure(net, p, rand_numbs):
    '''
    Computes complexity of output based on squared second derivatives.
    This measure is inspired by smoothing splines, where they "penalize" models for being 
    "too complex" (which is measures through the sum of second derivatives).

    We also include the first derivatives. Reason for this is that we need to know 
    wether variable is contributing to the output at all. If both first and second 
    derivative "measures" are equal to zero, then we know that the input in question 
    does not contribute to the output (as it is either a constant or zero)

    Other than this, we know at least the following:
        - complexity for an input equal to zero --> linear contribution to prediction
        - complexity for an input greater than zero --> non-linear realationship
        - combined complexity equal to zero --> linear contribution from inputs (at least some of the inputs)
        - combined complexity greater than zero --> non-linear function
    
    It should also be noted that this complexity measure is a way of determining which
    models that are equivalent, and which models that are "more complex" than others. 
    Sometimes it is very easy to determine wether a trained model is "more complex" than 
    another. However, in some situations, it is not that obvious, meaning a measure like this
    would be really helpful. This metric could also "help" when considering multiple models
    with similar accuacy and similar density. Also, it could help us get a better understanding 
    of the "underlying" function of the model. That is, we could understand the "complexity" of 
    the underlying function, and which inputs that has the most "complex" measures.
    '''
    first, second = second_derivative_output(net, p, rand_numbs)
    mc_int_first = {}  # First derivative 
    mc_int_second = {}  # Second derviative "complexity"
    for i in range(p):
        mc_int_first[i] = np.mean(np.array(first[i])**2)  # Also sqaure first derivative to avoide "canceling" out pos and neg values
        mc_int_second[i] = np.mean(np.array(second[i])**2)  # This is the complexity for each input

    combined_complexity = np.sum([k for k in mc_int_second.values()])  # The "cumulative" complexity

    return mc_int_first, mc_int_second, combined_complexity

def train(net,train_data, optimizer, batch_size, num_batches, p, DEVICE, nr_weights, flows=False, reg=False, verbose=True, epoch=1):
    net.train()
    old_batch = 0
    for batch in range(int(np.ceil(train_data.shape[0] / batch_size))):
        batch = (batch + 1)
        _x = train_data[old_batch: batch_size * batch,0:p]
        _y = train_data[old_batch: batch_size * batch, -1]
        
        old_batch = batch_size * batch
        data = _x.to(DEVICE)
        target = _y.to(DEVICE)
        # print(target)
        target = target.unsqueeze(1).float()
        # print(target)
                
        net.zero_grad()
        if flows:
            outputs = net(data, sample=True)
            # print(outputs)
            negative_log_likelihood = net.loss(outputs, target)
            # negative_log_likelihood = F.nll_loss(outputs, target, reduction="sum")
            # if epoch <= 1000:
            #     loss = negative_log_likelihood + (epoch/1000)*(net.kl() / num_batches)
            # else:
            loss = negative_log_likelihood + net.kl() / num_batches
        else:
            loss, log_prior, log_variational_posterior, negative_log_likelihood = net.sample_elbo(data, target)
        loss.backward()
        optimizer.step()
    
    #if val_data != None:
    #    print("Val data:")
    #    test_ensemble(net, val_data, DEVICE, SAMPLES=10, flows=flows, reg=reg, verbose=True)
    if verbose:
        print('loss', loss.item())
        print('nll', negative_log_likelihood.item())
        print('density', expected_number_of_weights(net, p)/nr_weights)
        print('')
    return negative_log_likelihood.item(), loss.item()

def val(net, val_data, DEVICE, flows=False, reg=False, verbose=True):
    '''
    NOTE: Will only validate using median model as this is 
            what we mainly care about. Reason for this is 
            that the full model could give missleading results
            as there are too many redundant weights that are 
            included
    '''
    net.eval()
    with torch.no_grad():
        _x = val_data[:, :-1]
        _y = val_data[:, -1]
        data = _x.to(DEVICE)
        target = _y.to(DEVICE)
        target = target.unsqueeze(1).float()
        if flows:
            outputs = net(data, ensemble=False)
            negative_log_likelihood = net.loss(outputs, target)
            # negative_log_likelihood = F.nll_loss(outputs, target, reduction="sum")
            loss = negative_log_likelihood + net.kl() 
        else:
            '''
            TODO: Fix this such that it is correct for "normal" LBBNN models...
            '''
            loss, log_prior, log_variational_posterior, negative_log_likelihood = net.sample_elbo(data, target)


        if reg:
            metric = R2Score()
            a = metric(outputs.T[0], target.T[0]).cpu().detach().numpy()
        else:
            output1 = outputs.T.mean(0)
            
            class_pred = output1.round().squeeze()
            # class_pred = output1.max(1, keepdim=True)[1]
            a = np.mean((class_pred.cpu().detach().numpy() == target.cpu().detach().numpy().T[0]) * 1)
            # a = class_pred.eq(target.view_as(class_pred)).sum().item() / len(target)
    
    alpha_clean = clean_alpha(net, threshold=0.5)
    density_median, used_weigths_median, _ = network_density_reduction(alpha_clean)
    if verbose:
        print(f'val_loss: {loss.item():.4f}, val_nll: {negative_log_likelihood.item():.4f}, val_ensemble: {a:.4f}, used_weights_median: {used_weigths_median}\n')

    return negative_log_likelihood.item(), loss.item(), a

def test_ensemble(net, test_data, DEVICE, SAMPLES, flows=False, reg=True, verbose=True):
    net.eval()
    metr = []
    metr_median = []
    density = []
    used_weights = []
    ensemble = []
    ensemble_median = []
    with torch.no_grad():
        _x = test_data[:, :-1]
        _y = test_data[:, -1]
        data = _x.to(DEVICE)
        target = _y.to(DEVICE)
        #outputs = torch.zeros(TEST_SAMPLES, TEST_BATCH_SIZE, 1).to(DEVICE)
        for _ in range(SAMPLES):
            if flows:
                outputs = net.forward(data, sample=True, ensemble=True)
                outputs_median = net.forward(data, sample=True, ensemble=False)
            else:
                for l in net.linears:
                    l.alpha = 1 / (1 + torch.exp(-l.lambdal))
                    l.gamma.alpha = l.alpha

                # sample the model
                cgamma = [l.gamma.rsample().to(DEVICE) for l in net.linears]
                outputs = net.forward(data, cgamma, sample=True)


            alpha_clean = clean_alpha(net, threshold=0.5)
            density_median, used_weigths_median, _ = network_density_reduction(alpha_clean)
            density.append(density_median)
            used_weights.append(used_weigths_median)

            if reg:
                metric = R2Score()
                a = metric(outputs.T[0], target).cpu().detach().numpy()
                a_median = metric(outputs_median.T[0], target).cpu().detach().numpy()
            else:
                output1 = outputs.T.mean(0)
                class_pred = output1.round().squeeze()
                a = np.mean((class_pred.cpu().detach().numpy() == target.cpu().detach().numpy()) * 1)

                output1_median = outputs_median.T.mean(0)
                class_pred_median = output1_median.round().squeeze()
                a_median = np.mean((class_pred_median.cpu().detach().numpy() == target.cpu().detach().numpy()) * 1)
            
            ensemble.append(a)
            ensemble_median.append(a_median)
            # get_metrics(net, threshold=0.5)

        metr.append(np.mean(ensemble))
        metr.append(np.mean(density))
        metr_median.append(np.mean(ensemble_median))
        metr_median.append(np.mean(used_weights))

    if verbose:
        print(np.mean(density), 'density median')
        print(np.mean(used_weights), 'used weights median')
        print(np.mean(ensemble), 'ensemble full')
        print(np.mean(ensemble_median), 'ensemble median')
        

    return metr, metr_median


