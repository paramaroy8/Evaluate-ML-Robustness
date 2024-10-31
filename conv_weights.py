import torch

import numpy as np



def compute_weights():
    # getting the raw weights from the layer
    raw_weights = torch.load("conv1_weights.pt",map_location=torch.device('cpu'))
    
    '''
    Initialize layer weights matrix that will be used to compute distance matrix and loss

    Dimensions of Layer Weight matrix:
            • each original image is 28 by 28
            • each filtered image is 26 by 26
            • there are total 32 filts
    '''

    torch_layer_weights = torch.zeros((32, 26, 26, 28, 28))
    
    # Process the weights by computing layer weights from the raw weights

    for i in range(32):
        for j in range(26):
            for k in range(26):
                for filter_row in range(3):
                    for filter_column in range(3):
                        torch_layer_weights[i][j][k][j+filter_row][k+filter_column] = raw_weights[i][0][filter_row][filter_column]
    
    layer_weights = torch_layer_weights.detach().cpu().numpy()
    
    return layer_weights

