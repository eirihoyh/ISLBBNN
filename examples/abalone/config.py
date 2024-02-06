#!/usr/bin/env python3
# -*- coding: utf-8 -*-

config = {}
config['num_epochs' ] = 100  
config['post_train_epochs'] = 50
config['n_nets'] = 1  # Number of different networks to run for one problem
config['class_problem'] = False  # If classification problem or not
config['n_layers'] = 2+2 # Four hidden + (one input+one output)
config['test_samples'] = 5000
config['lr'] = 0.01
config['num_transforms'] = 2  # For normalizing flows
config['hidden_dim'] = 20 # Reduced to make it quicker to run simple checks

config['verbose'] = True  # If we want printouts during/after training or not
config['save_res'] = False  # If we should save the results
config['patience'] = 10000