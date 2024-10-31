'''
This is the python equivalent of euclidean_distance.ipynb file except we will not use euclidean distance for anything. 
This file will only be used to process the MNIST dataset and collect the class examples. 
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance

# read dataset
df = pd.read_csv('mnist_train_data.csv')
    
# get all labels in the dataframe
labels = df['label']

# creating new dataset without labels
dataset = df.drop('label', axis = 1)

def get_class_data(class_name, total_samples):
    mat = []
    total_count = 0  # total number of examples of the given class among the samples
    
    for i in range(0, total_samples):
        if labels[i] == class_name:
            mat.append(dataset.iloc[i])
            total_count += 1
     
    n_mat = np.array(mat)
    
    return n_mat, total_count

def process_dataset():
#     # read dataset
#     df = pd.read_csv('mnist_train_data.csv')
    
#     # get all labels in the dataframe
#     labels = df['label']
    
#     # creating new dataset without labels
#     dataset = df.drop('label', axis = 1)
    
    # class_1 = all values related to the class named 3
    # class_2 = all values related to the class named 7

    class_1, class_1_count = get_class_data(3, 42000)
    class_2, class_2_count = get_class_data(7, 42000)
    
    '''
    both classes have different number of samples available to compute distance, we will need a certain     number of samples that are available for both classes
 
    iter_count variable keeps that count
    '''
    iter_count = class_1_count if class_1_count < class_2_count else class_2_count
    
    
    # return varied class 2 examples for each set of class 1 examples
    return class_1, class_2[:iter_count]
    
    
    
