def load_packages():
    # General imports
    import subprocess
    import sys


    def install(package):
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

    install('pandas')
    install('numpy')
    install('scipy')
    install('scikit-learn')

load_packages()

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
import math
from sklearn.model_selection import train_test_split
import pickle
import os
import h5py

################################################### Data Processing ########################################################################

def split_data(feature_data, target_data,data_size, test_size=1000, val_size=0.2):
    """ Split the data into training, validation and test sets

    Parameters
    ---------
    feature_data : np.array
        data with only features
    target_data : np.array
        data with only target
    test_size : float
        proportion of the dataset to include in the test split
    val_size : float
        proportion of the dataset to include in the validation split

    Returns
    ---------
    X_train : np.array
        training feature data
    y_train : np.array
        training target data
    X_val : np.array
        validation feature data
    y_val : np.array
        validation target data
    X_test : np.array
        test feature data
    y_test : np.array
        test target data
    """
    # First, split the data into training+validation set and test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(feature_data, target_data, test_size=test_size, shuffle=False)

    # Select the last data_size elements for training and validation
    X_train_val = X_train_val[-data_size:]
    y_train_val = y_train_val[-data_size:]

    # Then, split the training+validation set into training set and validation set
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size, random_state=42)

    return X_train, y_train, X_val, y_val, X_test, y_test


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
    np.random.seed(42)
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
    np.random.seed(42)
    Theta1 = np.random.randn(num_dimensions, num_dimensions) * 0.01
    Theta2 = np.random.randn(num_dimensions, num_dimensions) * 0.01

    Theta1[:3, :3] = base_Theta1 * factor
    Theta2[:3, :3] = base_Theta2 * factor

    return Theta1 , Theta2 



def get_A_B(target_size=6):
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
    return A[:target_size,:], B[:target_size,:]


################################################### Data Generation Functions ########################################################################

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
    X[0] = [1,1,1]
    X[1] = [1,1,1]
    for t in range(2, n_periods):
        # Calculate the current value based on past values and innovations of the last two periods
        X[t] = Phi1 @ X[t-1] + Phi2 @ X[t-2] + U[t] + Theta1 @ U[t-1] + Theta2 @ U[t-2]
    return X

def generate_demand(X, n_periods, target_size):
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
    
    A, B = get_A_B(target_size)
    # create noise term epsilon

    X = X + 100
    print(X[0:5,:])

    np.random.seed(42)
    epsilon = np.random.normal(0, 10, (n_periods, target_size))
    # create noise term delta
    np.random.seed(24)
    delta = np.random.normal(0, 10, (n_periods, 3)) / 4
    # transform the input features X
    Y = np.maximum(0, (X + delta) @ A.T + (X @ B.T) * epsilon) 

    Y = np.round(Y, 0)    
    
    return Y

import numpy as np

def add_info_features(data, L=1):
    """
    Add 3L dimensions to the dataset by creating L copies of each feature and adding Gaussian noise.
    
    Parameters:
        data (np.array): The original dataset, assumed to be an array where rows are samples and columns are features.
        L (int): The number of copies to be created for each feature.
        
    Returns:
        np.array: The expanded dataset with additional noisy features.
    """    
    expanded_data = data.copy()
    np.random.seed(42)

    # Generate new features by adding Gaussian noise to copies of the original features
    for i in range(L):
        
        i = 64 - i
        # Compute the scale of the noise
        #sigma = (1/2) ** (i)  
        sigma =  (i/10)

        # Add Gaussian noise scaled by sigma to each feature
        noisy_features = data +  np.random.normal(-sigma, sigma, data.shape) 
        
        # Place the noisy features in the corresponding columns of the new feature array
        expanded_data = np.append(expanded_data, noisy_features, axis=1)
    
    return expanded_data

def add_noise_features(data, amount=1):
    """
    Add L dimensions to the dataset by creating L noise features with the same distribution as the original features.
    
    Parameters:
        data (np.array): The original dataset, assumed to be an array where rows are samples and columns are features.
        L (int): The number of noise features to be created.
        
    Returns:
        np.array: The expanded dataset with additional noise features.
    """    
    expanded_data = data.copy()
    np.random.seed(42)

    # Generate new features by creating noise with the same distribution as the original features
    for _ in range(amount):
        # Compute the mean and standard deviation of the data
        mean = np.mean(data)
        std = np.std(data)

        # Generate noise with the same distribution as the data
        noise_features = np.random.normal(mean, std, data.shape) 
        
        # Append the noise features to the dataset
        expanded_data = np.append(expanded_data, noise_features, axis=1)
    
    return expanded_data

def add_heterogenity(data:np.array, factor:float, percentage:float):
    """ Add heterogenity to the dataset by multiplying a random subset of the data with a factor

    Parameters:
    -----------
    data : np.array
        dataset to add heterogenity to
    factor : float
        scale between 0 and 1 that indicates the heterogenity of the dataset

    Returns:
    -----------
    data : np.array
        dataset with added heterogenity
    hetero_bool : np.array
        boolean array that indicates which datapoints have been multiplied with the factor
    """
    data_size = data.shape[0] # number of datapoints
    np.random.seed(42) # set seed for reproducibility
    
    # Create Boolean array to indicate which datapoints are multiplied with the factor
    hetero_bool = np.random.choice([0, 1], size=(data_size, 1), p=[(1-percentage), percentage]) 

    # Multiply the datapoints with the factor
    for i in range(data_size):
        if hetero_bool[i] == 1:
            data[i,:] = data[i,:] * (1+(factor))#/10)
        
    return data, hetero_bool


################################################### Data Generation ########################################################################


def generate_data(data_size:int, feature_size:int, feature_use:bool, target_size:float, volatility:float, heterogenity:float, path):
    """ Generate an artificial dataset based on the settings and saves it 
    
    Parameters:
    -----------
    data_size : int
        indicates size of datapoints (10^1, 10^2, ..., 10^6)^
        default = 10^5
    feature_size : int
        indicates count of features (3, 6, 12, 24, ..., 384)
        default = 12
    feature_use : boolean
        indicates whether additional features give additional info about the target, or they are just noise
        default = False
    target_size : int
        indicates count of features 
        default = 6
    volatility : float
        scale between 0 and 1 that indicates the volatility
        default = 0.05
    heterogenity : float
        scale between 0 and 1 that indicates the heterogenity of the dataset
        default = 0
    path : str
        path to save the data
        
    Returns:
    -----------
    dataset_id : str
        unique identifier for the dataset
    file_path : str
        path to the saved dataset
    
    """
    dimensions = 3
    set_length = 10**6 + 100 # add 100 as test size

    """
    generate the covariance matrix for the innovations U
    - diagonal: increasing the scale_diagonal increases the variance of the innovations --> more volatile
    - off-diagonal: increasing the scale_off_diagonal increases the correlation between the innovations --> more synchronized
    """
    sigma_U = generate_sigma_U(num_dimensions=dimensions, scale_diagonal=volatility, scale_off_diagonal=0.05)
    

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

    # simulate the ARMA process
    X = simulate_arma22(n_periods=set_length, Phi1=Phi1, Phi2=Phi2, Theta1=Theta1, Theta2=Theta2, sigma_U=sigma_U, num_dimensions=dimensions)

    # Handle additional Features
    if feature_use == True:
        # Add L additional copies of the orignial features, that have add each time a bit more info
        X = add_info_features(data=X, L=int(feature_size/3))
        # Drop the original features
        X = X[:,3:]
    elif feature_use == False and feature_size > dimensions:
        # Add noise-features
        X = add_noise_features(data=X, amount=(feature_size-dimensions))

    # generate the demand with the factor model
    Y = generate_demand(X=X[:,:3], n_periods=set_length, target_size=(target_size+3))
    Y = Y[:,3:]

    # Handle heterogenity and add boolean feature to indicate heterogenity
    # Define heterogenity info
    hetero_info = True

    print(heterogenity)
    if heterogenity > 0:
        match round(heterogenity,1):
            case 0.1: factor = 0.4
            case 0.2: factor = 0.55
            case 0.3: factor = 0.6
            case 0.4: factor = 0.75
            case 0.5: factor = 1.05
            case 0.6: factor = 1.5
        #factor = 0.3 + (heterogenity * 1.5)
        Y, hetero_bool = add_heterogenity(data=Y, factor=factor, percentage = heterogenity)

        if (hetero_info == True):
            X = np.append(X, hetero_bool, axis=1)

    # train, val test split
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(feature_data=X, target_data=Y, val_size=0.2, test_size=100, data_size=data_size)

    # create dataset ID and dir for the data
    dataset_id = "set_" + str(int(math.log10(data_size))) + str(int(feature_size)) + str(int(feature_use)) + str(int(target_size)) + str(int(volatility*100)) + str(int(heterogenity*100))
    if(hetero_info == False):
        dataset_id = dataset_id + "no"
    final_path = path +"/"+ dataset_id
    os.makedirs(final_path, exist_ok=True)

    # create file path and save the data
    file_path = final_path + "/" + dataset_id +"_data.h5"
    print(path)
    print(file_path)
    with h5py.File(file_path, 'w') as f:
        f.create_dataset('X_train', data=X_train)
        f.create_dataset('y_train', data=y_train)
        f.create_dataset('X_val', data=X_val)
        f.create_dataset('y_val', data=y_val)
        f.create_dataset('X_test', data=X_test)
        f.create_dataset('y_test', data=y_test)
    
    # pickle dataset
    return {'dataset_id': dataset_id, 'dataset_path': file_path, 'folder_path': final_path}


#################################################################### Main ####################################################################

if __name__ == "__main__":

    save_path = "/pfs/work7/workspace/scratch/ma_elanza-thesislanza" #"C:/Users/lanza/Master_Thesis_EL/Integrated-vs-Seperated-Master-Thesis/test"

    dataset_list = []

    # Try to load existing data
    dataset_file = save_path + "/dataset_list.pkl"
    if os.path.exists(dataset_file):
        with open(dataset_file, 'rb') as f:
            dataset_list = pickle.load(f)

    # Create data for different sizes (10 - 1.000.000)
    for i in range(7):
        
        dataset_dict = generate_data(data_size=(10**4), feature_size=3, feature_use=False, target_size=6, volatility=(0.05), heterogenity=0, path=save_path)
        dataset_list.append(dataset_dict)


        # Info Features (3* (2**i))     Default 3
        # Heterogenity (0.1 * i)        Default 0

    # Write the updated list back to the file
    with open(dataset_file, 'wb') as f:
        pickle.dump(dataset_list, f)
