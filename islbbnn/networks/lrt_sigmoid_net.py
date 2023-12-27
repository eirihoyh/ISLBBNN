import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.lrt_layers import BayesianLinear

class BayesianNetwork(nn.Module):
    def __init__(self, dim, p, hidden_layers, a_prior=0.05, classification=True, n_classes=1):
        '''
        TODO: Add option to select perfered loss self wanting to test another loss type 
        '''
        super().__init__()
        self.p = p
        self.classification = classification
        self.multiclass = n_classes > 1
        # set the architecture
        self.linears = nn.ModuleList([BayesianLinear(p, dim, a_prior=a_prior)])
        self.linears.extend([BayesianLinear((dim+p), (dim), a_prior=a_prior) for _ in range(hidden_layers-1)])
        self.linears.append(BayesianLinear((dim+p), n_classes, a_prior=a_prior))
        if classification:
            if not self.multiclass: # For multiclass, F.nll_loss is used in the training loop
                self.loss = nn.BCELoss(reduction='sum') # Setup loss (Binary cross entropy as binary classification)
        else:
            self.loss = nn.MSELoss(reduction='sum')     # Setup loss (Mean Squared Error loss as regression problem)
            
    def forward(self, x, sample=False, ensemble=True, post_train=False):
        x_input = x.view(-1, self.p)
        x = F.sigmoid(self.linears[0](x_input, ensemble, sample, post_train=post_train))
        i = 1
        for l in self.linears[1:-1]:
            x = F.sigmoid(l(torch.cat((x, x_input),1), ensemble, sample, post_train=post_train))
            i += 1

        if self.classification:
            if self.multiclass:
                out = F.log_softmax((self.linears[i](torch.cat((x, x_input),1), ensemble)), dim=1)
            else:
                out = torch.sigmoid(self.linears[i](torch.cat((x, x_input),1), ensemble, sample, post_train=post_train))
        else:
            out = self.linears[i](torch.cat((x, x_input),1), ensemble, sample)
        return out

    def kl(self):
        kl_sum = self.linears[0].kl
        for l in self.linears[1:]:
            kl_sum = kl_sum + l.kl
        return kl_sum
    
