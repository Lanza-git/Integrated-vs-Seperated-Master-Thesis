import numpy as np
import h5py
import os
import pickle
from scipy.stats import chi2

def load_metadata(directory):
    metadata = []
    for file_name in os.listdir(directory):
        if file_name.endswith('meta.pkl'):
            with open(os.path.join(directory, file_name), 'rb') as f:
                data = pickle.load(f)
                metadata.append({
                    'file_name': file_name,
                    'hyperparameter': data['hyperparameter'],
                    'profit': data['profit'],
                    'elapsed_time': data['elapsed_time'],
                    'peak_memory' : data['peak_memory'],
                    'avg_memory': data['avg_memory']
                })
    return metadata

def load_generated_data(path:str, multi:bool=True):
    """ Load the generated data for the newsvendor problem from specified location 

    Parameters
    ---------
    path : path to the data file
    
    Returns
    ---------
    raw_data : pd.dataframe
    """ 
    # Load Data
    with h5py.File(path, 'r') as file:
        X_train = file['X_train'][:]
        y_train = file['y_train'][:]
        X_val = file['X_val'][:]
        y_val = file['y_val'][:]
        X_test = file['X_test'][:]
        y_test = file['y_test'][:]
    
    if multi == False:
        y_train = y_train[:,0]
        y_val = y_val[:,0]
        y_test = y_test[:,0]

    return X_train, y_train, X_val, y_val, X_test, y_test

def load_cost_structure(alpha_input:np.array, underage_input:np.array, overage_input:np.array):
    """ Initialize the cost structure for the newsvendor problem"""
    global alpha, underage, overage
    alpha = alpha_input
    underage = underage_input
    overage = overage_input

def nvps_profit(demand:np.array, q:np.array):
    """ Profit function of the newsvendor under substitution
    
    Parameters
    ---------
    demand : actual demand, shape (T, N_PRODUCTS)
    q : predicted orders, shape (T, N_PRODUCTS)

    Returns
    ---------
    profits: np.array
        Profits by period, shape (T,1)
    """
    global alpha, underage, overage

    # Round values
    demand = np.round(demand, 0)
    q = np.round(q, 0)

    # Check if the array is 1D
    if demand.ndim == 1:
        # Reshape the array to have shape (n, 1)
        demand = demand.reshape(-1, 1)
    # Check if the array is 1D
    q = np.array(q,copy=False)
    if q.ndim == 1:
        q = q.reshape(-1, 1)# Reshape the array to have shape (n, 1)

    if demand.shape[1] == 1:
        q = np.maximum(0., q)
        if len(q.shape) == 1:
            q = q.reshape(-1, 1)
        profits = np.sum(np.minimum(q,demand)*underage - np.maximum(demand-q, 0.)*(underage)-np.maximum(q-demand, 0.)*(overage))
    else:
        q = np.maximum(0., q) # make sure orders are non-negative
        demand_s = demand + np.matmul(np.maximum(demand-q, 0.), alpha) # demand including substitutions
        profits = np.sum(np.matmul(q, underage) - np.matmul(np.maximum(q-demand_s, 0.), (underage+overage))) # period-wise profit (T x 1)
    return profits

def calculate_cochran(X_test, y_test):
    # Before calculating means and variances, check if x1 and x2 are empty
    if len(y_test) == 0 or len(X_test) == 0:
        print ("Empty")
        return np.nan, np.nan, np.nan  # Or handle this case as appropriate
    

    x1 = []
    x2 = []

    for i in range(len(X_test)):
        if X_test[i, -1] == 1:  # Check if the last column is 1
            x1.append(y_test[i])
        else:
            x2.append(y_test[i])

    # Convert lists to numpy arrays if needed
    x1 = np.array(x1)
    x2 = np.array(x2)
    
    # Calculate means and variances for each study
    mean_x1 = np.mean(x1)
    variance_x1 = np.var(x1, ddof=1)

    mean_x2 = np.mean(x2)
    variance_x2 = np.var(x2, ddof=1)

    # Combine means and variances
    means = np.array([mean_x1, mean_x2])
    variances = np.array([variance_x1, variance_x2])

    # Step 1: Calculate weights (inverse of variances)
    weights = 1 / variances

    # Step 2: Calculate the overall weighted effect
    weighted_effect = np.sum(weights * means) / np.sum(weights)

    # Step 3: Calculate Cochran's Q
    Q = np.sum(weights * (means - weighted_effect) ** 2)

    # Step 4: Calculate I²
    k = len(means)
    I2 = max(0, (Q - (k - 1)) / Q) * 100

    # Step 5: Calculate p-value for Q
    p_value = 1 - chi2.cdf(Q, df=k-1)

    return Q, I2, p_value
    

def get_constants(risk_factor):
    # Parameters for multi-item newsvendor problem
    prices = np.array([0.3, 0.5, 0.6, 0.5, 0.5, 0.5]) #price data
    prices = prices.reshape(6,1)
    costs = np.array([0.06, 0.06, 0.06, 0.06, 0.06, 0.06]) #cost data
    costs = costs.reshape(6,1)
    costs = costs * risk_factor
    salvages = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01]) #salvage data
    salvages = salvages.reshape(6,1)
    underage_data = prices - costs 
    overage_data = costs - salvages 
    underage_data_single = underage_data[0,0]
    overage_data_single = overage_data[0,0]



    alpha_data = np.array([             #alpha data
        [0.0, 0.11, 0.15, 0.14, 0.14, 0.15],
        [0.22, 0.0, 0.08, 0.11, 0.12, 0.12],
        [0.24, 0.07, 0.0, 0.07, 0.08, 0.07],
        [0.26, 0.09, 0.07, 0.0, 0.12, 0.12],
        [0.19, 0.10, 0.10, 0.12, 0.0, 0.14],
        [0.17, 0.13, 0.11, 0.11, 0.13, 0.0]
    ])
    return underage_data, overage_data, alpha_data, underage_data_single, overage_data_single


