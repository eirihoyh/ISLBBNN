#!/usr/bin/env python3
# -*- coding: utf-8 -*-

config = {}
config['num_epochs' ] = 2000  
config['post_train_epochs'] = 500
config['n_nets'] = 1  # Number of different networks to run for one problem
config['class_problem'] = True  # If classification problem or not
config['n_layers'] = 2+2 # Four hidden + (one input+one output)
config['lr'] = 0.05
config['a_prior'] = 0.05
config['num_transforms'] = 2  # For normalizing flows
config['hidden_dim'] = 50 # Reduced to make it quicker to run simple checks

config['verbose'] = True  # If we want printouts during/after training or not
config['save_res'] = False  # If we should save the results
config['patience'] = 10000