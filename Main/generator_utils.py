import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal


################################################### Coefficients ########################################################################

def generate_phi(num_dimensions=3, factor=1):
    """
    Set up the AutoRegressive coefficents (AR)

    Influence: How much past values of the time series affect the current value.
    - decrease: make data more random / unpredictable
    - increase: make data more predictable

    Input:
    ------------
    factor: scale factor for the coefficients
    num_dimensions: number of dimensions for the ARMA process

    Output:
    ------------
    Phi1: AR(1) coefficients
    Phi2: AR(2) coefficients
    """

    base_Phi1 = np.array([
        [0.5, -0.9, 0],
        [1.1, -0.7, 0],
        [0, 0, 0.5]
    ])

    base_Phi2 = np.array([
        [0, -0.5, 0],
        [-0.5, 0, 0],
        [0, 0, 0]
    ])

    # Extend base matrices to num_dimensions with small random values for new dimensions
    Phi1 = np.random.randn(num_dimensions, num_dimensions) * 0.01
    Phi2 = np.random.randn(num_dimensions, num_dimensions) * 0.01

    Phi1[:3, :3] = base_Phi1 * factor
    Phi2[:3, :3] = base_Phi2 * factor

    return Phi1 , Phi2 


def generate_theta(num_dimensions=3, factor=1):
    """
    Set up the Moving Average coefficients (MA)

    Influence: How much past error terms (unexpected shocks) affect the current value.
    - increse: make data more dependent on past shocks/errors, make data more volatile 
    - decrease: less sensitve to shocks

    Input:
    ------------
    factor: scale factor for the coefficients
    num_dimensions: number of dimensions for the ARMA process

    Output:
    ------------
    Theta1: MA(1) coefficients
    Theta2: MA(2) coefficients
    """

    base_Theta1 = np.array([
        [0.4, 0.8, 0],
        [-1.1, -0.3, 0],
        [0, 0, 0]
    ])

    base_Theta2 = np.array([
        [0, -0.8, 0],
        [-1.1, 0, 0],
        [0, 0, 0]
    ])

    # Extend base matrices to num_dimensions with small random values for new dimensions
    Theta1 = np.random.randn(num_dimensions, num_dimensions) * 0.01
    Theta2 = np.random.randn(num_dimensions, num_dimensions) * 0.01

    Theta1[:3, :3] = base_Theta1 * factor
    Theta2[:3, :3] = base_Theta2 * factor

    return Theta1 , Theta2 



def get_A_B():
    """
    A: This is a diagonal matrix that scales each dimension of X by 2.5.
    A is the mean-dependence of the ith demand on these factors with some idiosyncratic noise
    """
    A = 2.5 * np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8]
    ])


    """
    B: This matrix introduces dependencies between different dimensions of X. 
    For example, each element of the output is influenced by other elements of the input.
    """
    B = 7.5 * np.array([
        [ 0, -1, -1],
        [-1,  0, -1],
        [-1, -1,  0],
        [ 0, -1,  1],
        [-1,  0,  1],
        [-1,  1,  0],
        [ 0,  1, -1],
        [ 1,  0, -1],
        [ 1, -1,  0],
        [ 0,  1,  1],
        [ 1,  0,  1],
        [ 1,  1,  0]
    ])
    return A, B


################################################### Functions ########################################################################

def generate_sigma_U(num_dimensions=3, scale_diagonal=0.05, scale_off_diagonal=0.05):
    """
    Generate the covariance matrix for the innovations U.

    - diagonal: increasing the scale_diagonal increases the variance of the innovations --> more volatile
    - off-diagonal: increasing the scale_off_diagonal increases the correlation between the innovations --> more synchronized

    Input: 
    ------------
    num_dimensions: number of dimensions for the covariance matrix
    scale_diagonal: scale factor for the diagonal elements
    scale_off_diagonal: scale factor for the off-diagonal elements

    Output:
    ------------
    sigma_U: covariance matrix for the innovations U
    """
    # Initialize the covariance matrix with zeros
    sigma_U = np.zeros((num_dimensions, num_dimensions))

    # Iterate over the matrix to set diagonal and off-diagonal elements
    for i in range(num_dimensions):
        for j in range(num_dimensions):
            if i == j:
                sigma_U[i, j] = (8/7) * scale_diagonal
            else:
                sigma_U[i, j] = (-1)**(i+j) * (1/7) * scale_off_diagonal

    return sigma_U

def generate_innovations(n_periods, sigma_U, num_dimensions=3):
    """
    Use NumPy to generate multivariate normal innovations.

    Innovations are the unpredictable part of the time series, representing new information or random shocks that occur at each time point. 
    They are the difference between the observed value and the value predicted by the model based on past values.
    """
    return multivariate_normal.rvs(mean=np.zeros(num_dimensions), cov=sigma_U, size=n_periods)

def simulate_arma22(n_periods, Phi1, Phi2, Theta1, Theta2, sigma_U, num_dimensions=3):
    """
    Simulation of a 3-dimensional ARMA(2,2) process.

    Input: number of periods
    Output: matrix where each row represents simulated values for a given time period   
    """
    # Initialize the matrix with zeros
    X = np.zeros((n_periods, num_dimensions))
    # Generate the innovations
    U = generate_innovations(n_periods=n_periods, sigma_U=sigma_U, num_dimensions=num_dimensions)

    # Simulate the ARMA process
    for t in range(2, n_periods):
        # Calculate the current value based on past values and innovations of the last two periods
        X[t] = Phi1 @ X[t-1] + Phi2 @ X[t-2] + U[t] + Theta1 @ U[t-1] + Theta2 @ U[t-2]
    return X

def generate_demand(X, n_periods):
    """
    Generate demand based on the 3-dimensional ARMA process and the factor model. 

    Input: 
    ------------
    X: matrix of simulated values for the ARMA process (n_periods, 3)

    Output: 
    ------------
    Y: matrix of demand values

    Variables (noise terms):
    ------------
    epsilon: noise term for the demand generation 
        --> increasing - resulting demand Y is more volatile
    delta: independent noise term. This noise term is added to X to introduce additional variability.
        --> increasing - input features X will have more variability before being transformed - more variability
    """
    A, B = get_A_B()
    # create noise term epsilon
    np.random.seed(42)
    epsilon = np.random.normal(0, 1, (n_periods, 12))
    # create noise term delta
    np.random.seed(24)
    delta = np.random.normal(0, 1, (n_periods, 3)) / 4
    # transform the input features X
    Y = np.maximum(0, (X + delta) @ A.T + (X @ B.T) * epsilon)        
    return Y

import numpy as np

def add_dimensions(data, L=1):
    """
    Add 3L dimensions to the dataset by creating L copies of each feature and adding Gaussian noise.
    
    Parameters:
        data (np.array): The original dataset, assumed to be an array where rows are samples and columns are features.
        L (int): The number of copies to be created for each feature.
        
    Returns:
        np.array: The expanded dataset with additional noisy features.
    """    
    expanded_data = data.copy()

    # Generate new features by adding Gaussian noise to copies of the original features
    for i in range(L):
        print(i)
        # Compute the scale of the noise
        sigma = (1/2) ** (i+1)  

        # Add Gaussian noise scaled by sigma to each feature
        noisy_features = data * sigma+ np.random.normal(0, 1, data.shape) 
        
        # Place the noisy features in the corresponding columns of the new feature array
        expanded_data = np.append(expanded_data, noisy_features, axis=1)
    
    return expanded_data



################################################### Main ########################################################################
