'''
We will calculate and show both EMD and sinkhorn methods in this file with 20 budgets.
'''

import ot
import ot.plot
import time
import matplotlib.pyplot as plt
import numpy as np

# import data processing file
from process_dataset import *

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


'''
In this function,
        • variable r represents each budget
        • compute cost matrix by dividing the euclidean distance matrix by budget, then taking
        the minimum with 1
        • compute gamma matrix using sinkhorn method
        • in sinkhorn method, first two parameters are a and b representing uniform
        distributions on samples
'''

def compute_cost_matrix(number_of_samples, budget, a, b):
    # compute cost matrix
    cost_matrix = np.minimum((scaled_e_dist[0:number_of_samples, 0:number_of_samples] / budget), 1)
    
    return cost_matrix


def gamma_emd(cost_matrix, a, b):

    # gamma matrix obtained by ot.emd() function

    emd_gamma_matrix = ot.emd(a, b, cost_matrix)

    emd_t_cost = transportation_cost(cost_matrix, emd_gamma_matrix)
    print(f'emd transportation cost = {emd_t_cost}')

    emd_t_loss = transportation_loss(emd_t_cost)
    print(f'\nemd transportation loss = {emd_t_loss}')

    return emd_gamma_matrix, emd_t_cost, emd_t_loss


def gamma_sinkhorn(cost_matrix, reg, a, b):    
    # compute gamma matrix
    gamma_matrix = ot.sinkhorn(a, b, cost_matrix, reg)

    return gamma_matrix

'''
In this function:
        • compute transportation cost by taking inner product (double contraction)
        between cost matrix and gamma matrix
        • np.tensordot() output is in array form. We converted that into a list
'''

def transportation_cost(cost_matrix, gamma_matrix):

    t_cost = np.array(np.tensordot(cost_matrix, gamma_matrix, axes = 2)).tolist()

    return t_cost


'''
In this function,
        • compute transportation loss by subtracting cost from 1, then dividing the result
        by 2
'''

def transportation_loss(t_cost):

    t_loss = (1 - t_cost) / 2

    return t_loss


def compute_sinkhorn(number_of_samples, budget_list, a, b, all_budgets):
    # store all used reg values
    reg_value_store = []
    
    
    
    for reg_value in range(reg_range[0], reg_range[1]):
            
            x = 0
        
            t_cost_store = []

            # store all transportation loss values
            t_loss_store = []
            
            print(f'\nreg value = {2**(-reg_value)}\n')
            for budget in range(budget_list[0], budget_list[1]):            
                
                # compute cost matrix
                cost_matrix = compute_cost_matrix(number_of_samples, budget, a, b)

                # compute cost matrix and sinkhorn gamma matrix
                gamma_matrix = gamma_sinkhorn(cost_matrix, (2**(-reg_value)), a, b)

                # compute transportation cost
                t_cost = transportation_cost(cost_matrix, gamma_matrix)
                print(f'sinkhorn transportation cost = {t_cost}')

                # compute tranportation loss
                t_loss = transportation_loss(t_cost)
                print(f'\nsinkhorn transportation loss = {t_loss}\n')

                # storing all transportation costs
                t_cost_store.append(t_cost)

                # storing all reg values
                reg_value_store.append(2**(-reg_value))

                # storing all tranportation losses
                t_loss_store.append(t_loss)

            '''
            Plotting graph here does not include all the points in the same graph
            '''

            # Create the plot where x axis = budgets, y axis = sinkhorn loss
            plt.plot(all_budgets, t_loss_store, marker='o', linestyle='-', label=(f'1/2^{reg_value}'))
    
    return
        

'''
In this function,
        • experiment with varying bugets and varying reg values
'''

def experiments(number_of_samples, budget_list, reg_range):

    # store emd loss values for plotting
    all_emd_loss = []

    all_budgets = np.arange(budget_list[0], budget_list[1]).tolist()
    
    # define uniform distributions on samples

    a, b = np.ones((number_of_samples, )) / number_of_samples, np.ones((number_of_samples, )) / number_of_samples
    
    # compute and plot sinkhorn method
    compute_sinkhorn(number_of_samples, budget_list, a, b, all_budgets)
    

    for budget in range(budget_list[0], budget_list[1]):

        print(f'\n\n\t\t\tbudget = {budget}\n')
        
        # compute cost matrix
        cost_matrix = compute_cost_matrix(number_of_samples, budget, a, b)
        

        # compute gamma matrix, transportation cost and transportation loss of ot.emd()
        emd_gamma_matrix, emd_t_cost, emd_t_loss = gamma_emd(cost_matrix, a, b)
        
        all_emd_loss.append(emd_t_loss)

    # Create the plot where x axis = budgets, y axis = emd loss
    plt.plot(all_budgets, all_emd_loss, marker='s', linestyle='-.', label=('emd'))


    return

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
    plt.savefig('both_4000.jpg')
    plt.close()  # Close the figure after saving


    
    
# experiment parameters: samples = 10, budget = [1, 21], reg range = [0, 10]

number_of_samples = 4000

budget_list = [1, 21]
reg_range = [0, 10]

# create the figure where values will be plotted
plt.figure(figsize=(16, 16)) 

experiments(number_of_samples, budget_list, reg_range)

show_plot(number_of_samples)