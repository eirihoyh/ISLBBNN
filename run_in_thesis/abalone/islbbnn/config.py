#!/usr/bin/env python3
# -*- coding: utf-8 -*-

config = {}
config['num_epochs' ] = 3000  
# config['num_epochs' ] = 10  
config['post_train_epochs'] = 0
config['n_nets'] = 10  # Number of different networks to run for one problem
config['class_problem'] = False  # If classification problem or not
config['n_layers'] = 2+2 # Four hidden + (one input+one output)
config['test_samples'] = 5000
config['lr'] = 0.005
config['a_prior'] = 0.25
config['num_transforms'] = 2  # For normalizing flows
config['hidden_dim'] = 200 # Reduced to make it quicker to run simple checks

config['verbose'] = True  # If we want printouts during/after training or not
config['save_res'] = True  # If we should save the results
config['patience'] = 1000