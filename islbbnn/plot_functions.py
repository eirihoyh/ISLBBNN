import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
import copy
from graphviz import Digraph
import pipeline_functions as pip_func


def plot_whole_path_graph(alpha_list, all_connections, save_path, show=True):
    dot = Digraph(f"All paths!")
    n_layers = len(alpha_list) + 1
    dim = alpha_list[0].shape[0]
    layer_list = pip_func.create_layer_name_list(n_layers)
    all_names = []
    for layer_ind, connection in enumerate(all_connections):
        for t, f in connection:
            if f >= dim: # Check if we use input in Hidden layer
                from_node = f"I_{f-dim}"
            else:
                from_node = f"{layer_list[layer_ind]}_{f}"
                
            if from_node not in all_names:  # Add from node as a used name
                dot.node(from_node)
                all_names.append(from_node)

            if t >= dim and layer_ind+1 < n_layers: # TODO: consider removing if statement
                to_node = f"I_{t-dim}"
            else:    
                to_node = f"{layer_list[layer_ind+1]}_{t}"

            if to_node not in all_names:  # Add to node as a used name
                dot.node(to_node)
                all_names.append(to_node)

            # connect from_node to to_node; from_node --> to_node. Label with connection prob
            dot.edge(from_node, to_node, label=f"α={alpha_list[layer_ind][t][f]:.2f}")
        
    dot.node(f"All paths", shape="Msquare")
    # dot.edges(edges)
    dot.format = 'png' # save as PNG file
    dot.strict = True  # Remove duplicated lines
    # print(dot.source)
    dot.render(save_path, view=show)


def plot_whole_path_graph_weight(weight_list, all_connections, save_path, show=True):
    dot = Digraph(f"All paths!")
    n_layers = len(weight_list) + 1
    dim = weight_list[0].shape[0]
    layer_list = pip_func.create_layer_name_list(n_layers)
    all_names = []
    for layer_ind, connection in enumerate(all_connections):
        for t, f in connection:
            if f >= dim:
                from_node = f"I_{f-dim}"
            else:
                from_node = f"{layer_list[layer_ind]}_{f}"
                
            if from_node not in all_names:
                dot.node(from_node)
                all_names.append(from_node)

            if t >= dim and layer_ind+1 < n_layers:
                to_node = f"I_{t-dim}"
            else:    
                to_node = f"{layer_list[layer_ind+1]}_{t}"

            if to_node not in all_names:
                dot.node(to_node)
                all_names.append(to_node)

            dot.edge(from_node, to_node, label=f"w={weight_list[layer_ind][t][f]:.2f}")
        
    dot.node(f"All paths", shape="Msquare")
    # dot.edges(edges)
    dot.format = 'png' # save as PNG file
    dot.strict = True
    # print(dot.source)
    dot.render(save_path, view=show)


def run_path_graph(net, threshold=0.5, save_path="path_graphs/all_paths_input_skip", show=True):
    # net = copy.deepcopy(net)
    alpha_list = pip_func.get_alphas(net)
    clean_alpha_list = pip_func.clean_alpha(net, threshold)
    all_connections = pip_func.get_active_weights(clean_alpha_list)
    plot_whole_path_graph(alpha_list, all_connections, save_path=save_path, show=show)

def run_path_graph_weight(net, threshold=0.5, save_path="path_graphs/all_paths_input_skip", show=True, flow=False):
    # net = copy.deepcopy(net)
    weight_list = pip_func.weight_matrices_numpy(net, flow=flow)
    clean_alpha_list = pip_func.clean_alpha(net, threshold)
    all_connections = pip_func.get_active_weights(clean_alpha_list)
    plot_whole_path_graph_weight(weight_list, all_connections, save_path=save_path, show=show)


def plot_model_vision_image(net, train_data, train_target, c=0, net_nr=0, threshold=0.5, thresh_w=0.0, save_path=None):
    '''
    NOTE: Works just for quadratic images atm, should probably generalize to prefered
            dim at a later point
    '''
    
    colors = ["white", "red"]
    cmap = mcolors.LinearSegmentedColormap.from_list("", colors)
    
    clean_a = pip_func.clean_alpha_class(net, threshold,class_in_focus=c)
    p = clean_a[0].shape[1]
    img_avg = np.zeros(p)

    w = pip_func.weight_matrices(net)[-1][c, -p*p:].detach().numpy()
    w = np.where(clean_a[-1][c,-p*p:].detach().numpy() == 1, w, 0)
    
    avg_c_img = train_data[train_target == c].mean(axis=0)

    fig, axs = plt.subplots(len(clean_a)+1, figsize=(10,10))
    
    for ind, ca in enumerate(clean_a):
        out = ca.shape[0]
        img_layer = np.zeros(p)
        for j in range(out):
            # img_layer += ca[j,-p:].detach().numpy()
            img_layer += np.where(np.abs(w) >= thresh_w, ca[j,-p:].detach().numpy(), 0)

        img_avg += img_layer
        axs[ind].imshow(avg_c_img, cmap="Greys", vmin=torch.min(avg_c_img), vmax=torch.max(avg_c_img))
        if np.sum(img_layer) > 0:
            im = axs[ind].imshow(img_layer.reshape((p,p)), cmap=cmap, alpha=0.5)#, vmin=min_max*-1, vmax=min_max*1)
        else:
            im = axs[ind].imshow(img_layer.reshape((p,p)), cmap=cmap, alpha=0.5, vmin=0, vmax=1)
            
        fig.colorbar(im, ax=axs[ind])
        axs[ind].set_title(f"Class {c}, Layer {ind}")
        axs[ind].set_xticks([])
        axs[ind].set_yticks([])
        

    # min_max = max(np.concatenate((img_pos, img_neg*-1)))
    min_max = max(np.concatenate((img_avg, img_avg*-1)))

    
    axs[ind+1].imshow(avg_c_img, cmap="Greys", vmin=torch.min(avg_c_img), vmax=torch.max(avg_c_img))
    im = axs[ind+1].imshow(img_avg.reshape((28,28)), cmap=cmap, alpha=0.5, vmin=0, vmax=min_max*1)
    axs[ind+1].set_title(f"Net: {net_nr} all layers")
    axs[ind+1].set_xticks([])
    axs[ind+1].set_yticks([])
    fig.colorbar(im, ax=axs[ind+1])
    plt.tight_layout()
    if save_path != None:
        plt.savefig(save_path)
    plt.show()


def plot_local_contribution_empirical(net, data, sample=True, median=True, n_samples=1, save_path=None):
    '''
    Empirical local explaination model. This should be used for tabular data as 
    images usually has too many variables to get a good plot
    '''
    mean_contribution, std_contribution = pip_func.local_explain_relu(net, data, sample=sample, median=median, n_samples=n_samples)
    for c in mean_contribution.keys():
        labels = [str(k) for k in mean_contribution[c].keys()]
        means = list(mean_contribution[c].values())
        errors = list(std_contribution[c].values())

        fig, ax = plt.subplots()

        ax.bar(labels, means, yerr=errors, align='center', alpha=0.5, ecolor='black', capsize=10)
        ax.set_ylabel('Contribution')
        ax.set_xticks(labels)
        ax.set_title(f'Empirical approach class {c}')

        if save_path != None:
            plt.savefig(save_path+f"_class_{c}")

        plt.show()


def plot_local_contribution_dist(net, data, sample=False, median=True, save_path=None):
    cont_class = pip_func.local_explain_relu_normal_dist(net, data, sample=sample, median=median)
    for c in cont_class.keys():
        labels = [str(k) for k in cont_class[c].keys()]
        means = [val[0] for val in cont_class[c].values()]
        errors = [val[1] for val in cont_class[c].values()]

        fig, ax = plt.subplots()

        ax.bar(labels, means, yerr=errors, align='center', alpha=0.5, ecolor='black', capsize=10)
        ax.set_ylabel('Contribution')
        ax.set_xticks(labels)
        ax.set_title(f'Distribution approach class {c}')

        if save_path != None:
            plt.savefig(save_path+f"_class_{c}")

        plt.show()

def get_metrics(net, threshold=0.5):
    net = copy.deepcopy(net)
    # alpha_list = get_alphas(net)
    # p = alpha_list[0].shape[1]
    clean_alpha_list = pip_func.clean_alpha(net, threshold)
    p = clean_alpha_list[0].shape[1]
    layer_names = pip_func.create_layer_name_list()
    # all_connections = get_active_weights(clean_alpha_list)


    density, used_weights, tot_weights = pip_func.network_density_reduction(clean_alpha_list)
    print(f"Used {used_weights} out of {tot_weights} weights in median model.")
    print(f"Density reduced to:\n{(density*100):.4f}%\n")

    expected_density = pip_func.expected_number_of_weights(net)
    print(f"Expected to use {expected_density:.2f} out of {tot_weights} weights in the full model.")
    print(f"Density reduced to:\n{((expected_density/tot_weights)*100):.4f}%\n")

    mean_path_length, length_list = pip_func.average_path_length(clean_alpha_list)
    print(f"Average path length in network:\n{mean_path_length:.2f}")
    include_inputs = pip_func.include_input_from_layer(clean_alpha_list)
    print("Following inputs have been included:")
    for i, include in enumerate(include_inputs):
        print(f"Layer {layer_names[i]}: {include}\t  -->\tNr of inputs used: {sum(include)}")

    prob_include_input = pip_func.input_inclusion_prob(net)
    print("\nExpected number of input nodes included from a given layer:")
    for i,j in zip(prob_include_input.keys(), prob_include_input.values()):
        print(f"{i} --> {j:.4f}")

    # exp_depth, exp_depth_net = expected_depth(net, p)
    # print("\nExpected depth of the nodes:")
    # for i in range(p):
        # print(f"I{i}: {exp_depth[i]}")
    #print(f"Expected depth of network:\n{np.mean(list(exp_depth.values())):.4f}")
    # print(f"Expected depth of network:\n{exp_depth_net:.4f}")
    #return density, mean_path_length

    print(pip_func.prob_width(net, p))


def save_metrics(net, threshold=0.5, path="results/all_metrics"):
    # net = copy.deepcopy(net)
    clean_alpha_list = pip_func.clean_alpha(net, threshold)
    p = clean_alpha_list[0].shape[1]
    layer_names = pip_func.create_layer_name_list()

    density, used_weights, tot_weights = pip_func.network_density_reduction(clean_alpha_list)
    mean_path_length, length_list = pip_func.average_path_length(clean_alpha_list)
    include_inputs = pip_func.include_input_from_layer(clean_alpha_list)
    # NOTE: BUG HERE!!! Should do something with it later (does not give correct output for expected depth input)
    # TODO: FIX IT!!!
    # exp_node_depth_median  = expected_depth_median(net, p, threshold)

    metrics_median = {}
    metrics_median["layer_names"] = layer_names
    metrics_median["tot_weights"] = tot_weights
    metrics_median["used_weights"] = used_weights.detach().numpy()
    metrics_median["density"] = density.detach().numpy()
    metrics_median["avg_path_length"] = mean_path_length
    # metrics_median["expected_depth_input"] = exp_node_depth_median
    metrics_median["include_inputs"] = include_inputs
    
    
    # Save median model dictionary using numpy
    np.save(path+"_median", metrics_median)


    expected_density = pip_func.expected_number_of_weights(net)
    prob_include_input = pip_func.input_inclusion_prob(net)
    # exp_depth, exp_depth_net = expected_depth(net, p)

    metrics_full = {}
    metrics_full["layer_names"] = layer_names
    metrics_full["tot_weights"] = tot_weights
    metrics_full["expected_nr_of_weights"] = expected_density
    metrics_full["density"] = expected_density/tot_weights
    metrics_full["expected_nr"] = prob_include_input
    # metrics_full["expected_depth_inputs"] = exp_depth
    # metrics_full["expected_depth_net"] = exp_depth_net
    metrics_full["width_prob"] = pip_func.prob_width(net, p)
    
    # Save full model dictionary using numpy
    np.save(path+"_full", metrics_full)

