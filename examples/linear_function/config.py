#!/usr/bin/env python3
# -*- coding: utf-8 -*-

config = {}
config['num_epochs' ] = 100  
config['post_train_epochs'] = 0
config['n_nets'] = 1  # Number of different networks to run for one problem
config['n_samples'] = 4*10**4
config['class_problem'] = True  # If classification problem or not
config['non_lin'] = False  # Wanting to create a non-linear or linear dataset 
config['n_layers'] = 1+2 # Two hidden + (one input+one output)
config['test_samples'] = 5000
config['lr'] = 0.1
config['num_transforms'] = 2  # For normalizing flows
config['hidden_dim'] = 50 # Reduced to make it quicker to run simple checks

config['verbose'] = True  # If we want printouts during/after training or not
config['save_res'] = False  # If we should save the results
config['patience'] = 10000