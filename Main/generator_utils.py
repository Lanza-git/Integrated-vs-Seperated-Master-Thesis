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
    [0.1, 0.1, 0.8]
])
A = A[:3]

"""
B: This matrix introduces dependencies between different dimensions of X. 
For example, each element of the output is influenced by other elements of the input.
"""
B = 7.5 * np.array([
    [ 0, -1, -1],
    [-1,  0,  1],
    [-1,  1,  0],
    [ 0, -1,  1],
    [-1,  0,  1],
    [ 1, -1,  0],
    [ 0,  1,  1],
    [ 1,  0,  1],
    [ 1,  1,  0]
])
B = B[:3]

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
    Y: matrix of demand values (n_periods, 3)

    Variables (noise terms):
    ------------
    epsilon: noise term for the demand generation (n_periods, 3)
        --> increasing - resulting demand Y is more volatile
    delta: independent noise term. This noise term is added to X to introduce additional variability.
        --> increasing - input features X will have more variability before being transformed - more variability
    
    """
    epsilon = np.random.normal(0, 1, (n_periods, 3))
    delta = np.random.normal(0, 1, size=(len(X), 3)) / 4 
    Y = np.maximum(0, (X + delta) @ A.T + (X @ B.T) * epsilon)
    return Y


################################################### Main ########################################################################

if __name__ == '__main__':

    dimensions = 9

    """
    generate the covariance matrix for the innovations U
    - diagonal: increasing the scale_diagonal increases the variance of the innovations --> more volatile
    - off-diagonal: increasing the scale_off_diagonal increases the correlation between the innovations --> more synchronized
    """
    sigma_U = generate_sigma_U(num_dimensions=dimensions, scale_diagonal=0.05, scale_off_diagonal=0.05)

    """
    Set up the AutoRegressive coefficents (AR) 

    Influence: How much past values of the time series affect the current value.
    - decrease: make data more random / unpredictable
    - increase: make data more predictable
    """
    Phi1, Phi2 = generate_phi(num_dimensions=dimensions, factor=1)

    """
    Set up the Moving Average coefficients (MA)

    Influence: How much past error terms (unexpected shocks) affect the current value.
    - increse: make data more dependent on past shocks/errors, make data more volatile
    - decrease: less sensitve to shocks
    """
    Theta1, Theta2 = generate_theta(num_dimensions=dimensions, factor=1)

    # define the number of periods to simulate
    n_periods = 100
    # simulate the ARMA process
    X = simulate_arma22(n_periods=n_periods, Phi1=Phi1, Phi2=Phi2, Theta1=Theta1, Theta2=Theta2, sigma_U=sigma_U, num_dimensions=dimensions)

    # generate the demand
    Y = generate_demand(X[:,:3], n_periods)

    print(X.shape, Y.shape)


    df_target = pd.DataFrame(Y, columns=['Demand1', 'Demand2', 'Demand3'])
    df_features = pd.DataFrame(X, columns=['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5', 'Feature6', 'Feature7', 'Feature8', 'Feature9'])

    variance_target = df_target.var()
    print(variance_target)

    print(df_target.head())
    print(df_features.head())

    
