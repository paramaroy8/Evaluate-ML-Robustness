'''
We will calculate and show EMD plots for euclidean and CNN distances. 
'''

import ot
import ot.plot
import time
import matplotlib.pyplot as plt
import numpy as np

# import data processing file
from process_dataset import *

# get feature_distance
from cnn_distance import *


class_1, class_2 = process_dataset()

print(f'class 1 shape = {class_1.shape}, class 2 shape = {class_2.shape}')

# calling scipy.spatial.distance.cdist function to compute euclidean distance between two class examples

e_dist = distance.cdist(class_1, class_2, 'euclidean')

# scaling euclidean distance

scaled_e_dist = e_dist / 255

# print(f'e dist = {e_dist[:3]}, scaled e_dist = {scaled_e_dist[:3]}')

# initialize number of samples

number_of_samples = 0

# initialize list of budgets

budget_list = []

# initialize list of reg values

reg_range = []


def compute_emd(number_of_samples, budget, a, b):
    
    # compute cost matrix
    cost_matrix = np.minimum((scaled_e_dist[0:number_of_samples, 0:number_of_samples] / budget), 1)

    # gamma matrix obtained by ot.emd() function
    emd_gamma_matrix = ot.emd(a, b, cost_matrix)

    emd_t_cost = np.array(np.tensordot(cost_matrix, emd_gamma_matrix, axes = 2)).tolist()
    print(f'emd transportation cost = {emd_t_cost}')

    emd_t_loss = (1 - emd_t_cost) / 2
    print(f'\nemd transportation loss = {emd_t_loss}')

    return emd_gamma_matrix, emd_t_cost, emd_t_loss


def euc_experiments(number_of_samples, budget_list, a, b):

    # store emd loss values for plotting
    all_emd_loss = []
    

    for budget in range(budget_list[0], budget_list[1]):

        print(f'\n\n\t\t\tbudget = {budget}\n')
        
        _, _, loss = compute_emd(number_of_samples, budget, a, b)
        
        all_emd_loss.append(loss)


    return all_emd_loss


def cnn_experiments(number_of_samples, the_distance_matrix, a, b):
    
    # store emd loss values for plotting
    all_emd_loss = []

    
    for budget in range(budget_list[0], budget_list[1]):
        
        print(f'\n\nbudget = {budget}\n\n')

        _, _, loss = compute_emd(number_of_samples, budget, a, b)
    
        all_emd_loss.append(loss)
    
    return all_emd_loss



'''
This function will be called to show the graph.
'''

def show_plot(number_of_samples):
    # Add titles and labels
    plt.title(f'Number of Samples = {number_of_samples}')   
    plt.xlabel('Budget')
    plt.ylabel('Loss')

    # Add a legend
    plt.legend()

    # Show the plot
    plt.grid(True)

    # save figure
    plt.savefig(f'a_{number_of_samples}.jpg')
    plt.close()  # Close the figure after saving
    
    

# experiment parameters: samples = 300, budget = [1, 21]

number_of_samples = 300
budget_list = [1, 21]

# define uniform distributions on samples
a, b = np.ones((number_of_samples, )) / number_of_samples, np.ones((number_of_samples, )) / number_of_samples 

# euclidean experiments
euc_losses = euc_experiments(number_of_samples, budget_list, a, b)

# CNN experiments
the_distance_matrix = get_feature_extractor_distance(number_of_samples, budget_list)
cnn_losses = cnn_experiments(number_of_samples, the_distance_matrix, a, b)

# plots

all_budgets = np.arange(budget_list[0], budget_list[1]).tolist()

# create the figure where values will be plotted
plt.figure(figsize=(16, 16))

# x axis = budgets, y axis = euclidean loss
plt.plot(all_budgets, euc_losses, marker='o', linestyle='-.', label=('Euclidean'))

# x axis = budgets, y axis = CNN loss
plt.plot(all_budgets, cnn_losses, marker='|', linestyle='--', label=('CNN')) 

show_plot(number_of_samples)

# save values as .npy
data = np.array([euc_losses, cnn_losses, the_distance_matrix], dtype = object)

# save data 
np.save(f'cnn_{number_of_samples}.npy', data)