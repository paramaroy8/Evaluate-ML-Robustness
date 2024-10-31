'''
In this file, CNN experiments with 20 budgets.
'''

from cnn_distance import *

import ot
import ot.plot
import time
import matplotlib.pyplot as plt
import numpy as np


def cnn_experiments(number_of_samples, the_distance_matrix):
    
    # define uniform distributions on samples
    a, b = np.ones((number_of_samples, )) / number_of_samples, np.ones((number_of_samples, )) / number_of_samples
    
    # store emd loss values for plotting
    all_emd_loss = []
    
    # store all emd cost
    all_emd_cost = []

    all_budgets = np.arange(budget_list[0], budget_list[1]).tolist()

    
    for budget in range(budget_list[0], budget_list[1]):
        
        print(f'\n\nbudget = {budget}\n\n')

        # compute cost matrix
        cost_matrix = np.minimum((the_distance_matrix[0:number_of_samples, 0:number_of_samples] / budget), 1)
    
        # compute emd gamma matrix
        emd_gamma_matrix = ot.emd(a, b, cost_matrix)

        # compute emd cost
        emd_t_cost = np.array(np.tensordot(cost_matrix, emd_gamma_matrix, axes = 2)).tolist()
        print(f'emd transportation cost = {emd_t_cost}')
        
        all_emd_cost.append(emd_t_cost)
        
        # compute emd loss
        emd_t_loss = (1 - emd_t_cost) / 2
        print(f'\nemd transportation loss = {emd_t_loss}')
    
        all_emd_loss.append(emd_t_loss)

    # Create the plot where x axis = budgets, y axis = emd loss
    plt.plot(all_budgets, all_emd_loss, marker='s', linestyle='-.', label=('emd'))  
    
    return all_emd_cost, all_emd_loss


'''
This function will be called to show the graph.
'''

def show_plot(number_of_samples):
    # Add titles and labels
    plt.title(f'Number of Samples = {number_of_samples}')
    plt.xlabel('Budget')
    plt.ylabel('Loss')

#     # Add a legend
#     plt.legend()

    # Show the plot
    plt.grid(True)
    plt.show()
    
    # save figure
    plt.savefig('cnn_100.jpg')
    plt.close()  # Close the figure after saving


number_of_samples = 100
budget_list = [1, 21]

# create the figure where values will be plotted
plt.figure(figsize=(16, 16)) 

the_distance_matrix = get_feature_extractor_distance(number_of_samples, budget_list)

all_costs, all_losses = cnn_experiments(number_of_samples, the_distance_matrix)

show_plot(number_of_samples)

# save values as .npy
data = np.array([the_distance_matrix, all_losses, all_costs], dtype = object)

# save data 
np.save('cnn_100.npy', data)