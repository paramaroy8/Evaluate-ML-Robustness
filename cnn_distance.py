'''
We will compute the feature distance of the first layer a CNN model.
'''

from cvxopt import solvers, matrix

solvers.options['show_progress'] = False

import numpy as np

# import data processing file
from process_dataset import *

# get the CNN weights
from conv_weights import *



def compute_qp(A, t):
    n2, n1 = A.shape
    
    # define identity matrix in G matrix
    I_in_G = np.identity(n2)
    
    # define identity matrix in P matrix
    I_in_P = np.identity(n1)
    
    # define A times t 
    A_times_t = np.matmul(A, t)
    
    # define q column vector
    q = (((np.block([[np.zeros((n1, 1))],
             [np.ones((n2, 1))]
             ])) * np.linalg.norm(t)) / np.linalg.norm(A_times_t, ord=1)) * 1  # we are using lanbda_out = 1
    
    
    h = np.block([[np.asmatrix(-A_times_t).T], 
                  [np.asmatrix(A_times_t).T]
                 ])
    
    
    # define G matrix
    G = np.block([[A, -I_in_G], 
                  [-A, -I_in_G]
                 ])
    
    # old P matrix is divided by 2 norm of t
    P = ((np.block([
    [I_in_P, np.zeros((n1, n2))], 
              [np.zeros((n2, n1)), np.zeros((n2, n2))]
            ])) / np.linalg.norm(t)) * 2
    
    qp_output = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h))
    
    f_dist = qp_output.get('primal objective')
    
    return f_dist


'''
Call this function to collect the feature extractor distance matrix for the given number
of inputs examples for each class.

This function takes the following parameters:
            • how many exampels(rows) of each class to work with
            
            • how many lambda_out values to check with
            
            • weight matrix A of the layer
'''

def get_feature_extractor_distance(number_of_samples, budget_list):
    
    class_1, class_2 = process_dataset()

    print(f'class 1 shape = {class_1.shape}, class 2 shape = {class_2.shape}')

    # Normalize the class values between 0 and 1.
    class_1  = class_1/255
    class_2 = class_2/255
    
    # get model weights
    layer_weights = compute_weights()
    
    # CNN weight matrix (we are checking for all filters)
    A = (layer_weights[31, :, :, :, :]).reshape(((26*26), (28*28)))
    
    # initialize feature_extractor_distance_matrix
    feature_extractor_distance_matrix = np.zeros((number_of_samples, number_of_samples))
    
    for class_1_row in range(number_of_samples):
            
        for class_2_row in range(number_of_samples):
                
            # compute the t vector for qp programming
            t = (class_1[class_1_row] - class_2[class_2_row]).flatten()
                
            feature_extractor_distance_matrix[class_1_row][class_2_row] = compute_qp(A, t)
    
    return feature_extractor_distance_matrix