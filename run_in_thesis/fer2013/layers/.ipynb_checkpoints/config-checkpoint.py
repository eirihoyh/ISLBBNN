#!/usr/bin/env python3
# -*- coding: utf-8 -*-

config = {}
config['num_epochs' ] = 15000
config['n_nets'] = 1
config['n_samples'] = 4*10**4
config['n_layers'] = 4+2 # Two hidden + (one input+one output)
# config['num_epochs' ] = 1000
config['test_samples'] = 5000
config['lr'] = 0.0001
config['num_transforms'] = 2  # For normalizing flows
config['hidden_dim'] = 1000
# config['hidden_dim'] = 20 # Reduced to make it quicker to run simple checks

config['verbose'] = True  # If we want printouts during/after training or not
config['save_res'] = True  # If we should save the results
config['patience'] = 10000
