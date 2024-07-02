# Standard library imports
import os
import pickle
import datetime
import logging
import itertools
from typing import Tuple
import math
import threading
import time
import h5py

# Third-party imports
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
import optuna
from optuna_integration.keras import KerasPruningCallback
from optuna.integration import XGBoostPruningCallback
import gurobipy as gp
from gurobipy import GRB
import xgboost as xgb
import psutil

# scikit-learn imports
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# TensorFlow and Keras imports
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
from keras.utils import get_custom_objects
from scikeras.wrappers import KerasRegressor
from tensorflow.keras.utils import Sequence

# Create global variables for the cost structure
alpha = []
underage = []
overage = []

# Set up logging
logger = logging.getLogger(__name__)

######################## Environment Setup Functions #####################################################################    

def create_environment():
    """ Create the environment for the newsvendor problem"""

    # Set the environment variables for Gurobi
    os.environ['GRB_LICENSE_FILE'] = '/pfs/data5/home/ma/ma_ma/ma_elanza/test_dir/gurobi.lic'
    
    #os.environ['TF_ENABLE_ONEDNN_OPTS']=0


######################## Data Handling Functions ############################################################

def load_dict(path:str):
    """ Load a dictionary from a file

    Parameters
    ---------
    path : path to the file

    Returns
    ---------
    dictionary : dict
        dictionary loaded from the file
    """
    path = path + "/dataset_list.pkl"
    with open(path, 'rb') as file:
        dictionary = pickle.load(file)
    return dictionary

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

def load_data(path:str, multi:bool=False):
    """ Load  data for the newsvendor problem from specified location 

    Parameters
    ---------
    path : path to the data file
    multi : if True, all products are considered, if False, only product 1 is considered
    
    Returns
    ---------
    raw_data : pd.dataframe
    """ 
    # Load Data
    raw_data = pd.read_csv(path)    

    # Select only one product if multi == False
    if multi == False:
        # Select only columns with product_1_demand or not demand (features)
        selected_columns = raw_data.columns[raw_data.columns.str.contains('product_1_demand') | ~raw_data.columns.str.contains('demand')]
        raw_data = raw_data[selected_columns]
    return raw_data

def preprocess_data(raw_data:pd.DataFrame):
    """ Preprocess the data for the newsvendor problem
    
    Parameters
    ---------
    raw_data : raw data

    Returns
    ---------
    feature_data: pd.dataframe
        data with only features
    target_data: pd.dataframe
        data with only target
    """

    # Split the data into feature and target data
    feature_columns = raw_data.columns[raw_data.columns.str.contains('demand') == False]
    feature_data = raw_data[feature_columns]
    target_columns = raw_data.columns[raw_data.columns.str.contains('demand')]
    target_data = raw_data[target_columns]

    # Define preprocessing for numeric columns (scale them)
    numeric_features = feature_data.select_dtypes(include=[np.number]).columns.tolist()
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    # Define preprocessing for categorical features (encode them)
    categorical_features = feature_data.select_dtypes(exclude=[np.number]).columns.tolist()
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)],
        remainder='passthrough')

    # Preprocessing on  data
    feature_data = preprocessor.fit_transform(feature_data)

    return feature_data, target_data

def split_data(feature_data, target_data, test_size=0.2, val_size=0.2):
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
    X_train_val, X_test, y_train_val, y_test = train_test_split(feature_data, target_data, test_size=test_size, random_state=42)

    # Then, split the training+validation set into training set and validation set
    val_size_adjusted = val_size / (1 - test_size)  # Adjust the validation size
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size_adjusted, random_state=42)

    # Convert to numpy arrays if they are pandas DataFrames
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.values
    if isinstance(X_val, pd.DataFrame):
        X_val = X_val.values
    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.values
    if isinstance(y_train, pd.DataFrame) or isinstance(y_train, pd.Series):
        y_train = y_train.values
    if isinstance(y_val, pd.DataFrame) or isinstance(y_val, pd.Series):
        y_val = y_val.values
    if isinstance(y_test, pd.DataFrame) or isinstance(y_test, pd.Series):
        y_test = y_test.values

    return X_train, y_train, X_val, y_val, X_test, y_test

######################### Newsvendor related Functions ####################################################################

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

def solve_MILP(d:np.array, n_threads:int=40):
    """ helper function that solves the mixed-integer linear program (MILP) of the multi-product newsvendor problem under substitution (cf. slides)
    
    Parameters
    -----------
    d : Demand samples of shape (n, N_PRODUCTS), where n denotes the number of samples
    alpha : Substitution rates, shape (N_PRODUCTS, N_PRODUCTS)
    u : underage costs, shape (1, N_PRODUCTS)
    o : overage costs, shape (1, N_PRODUCTS)
    n_prods : number of products
    n_threads : number of threads

    Returns
    ----------
    orders: np.array
        Optimal orders, of shape (1, N_PRODUCTS)
    model.status : int
        Gurobi status code (see https://www.gurobi.com/documentation/current/refman/optimization_status_codes.html)
    """
    global alpha, underage, overage
    n_prods = d.shape[1] # number of products

    # Transpose 
    u = underage.T
    o = overage.T

    # Check the shapes of d and alpha
    if d.shape[1] != alpha.shape[0]:
        d = d.T

    hist = d.shape[0] # number of demand samples 
    d_min = np.min(d, axis=0) # minimum demand for each product
    d_max = d + np.matmul(d, alpha) # maximum demand for each product
    M = np.array(np.max(d_max, axis=0)[0]-d_min) # big M

    # intialize model, disable console logging and set number of threads
    model = gp.Model()
    model.Params.LogToConsole = 0
    model.Params.Threads = n_threads

    # initialize model variables
    z = model.addVars(hist, n_prods, vtype=GRB.BINARY)
    q = model.addVars(n_prods)
    y = model.addVars(hist, n_prods)
    v = model.addVars(hist, n_prods)

    # objective function
    obj = gp.LinExpr()
    for i in range(n_prods):
        obj += u[0, i].item()*q[i]
        for t in range(hist):
            obj -= (u[0, i].item()+o[0, i].item()) / hist * y[t, i]
    model.setObjective(obj, GRB.MAXIMIZE)

    # constraints
    for i in range(n_prods):
        for t in range(hist):
            model.addConstr(y[t, i]>=q[i]-d[t, i].item()-gp.quicksum(alpha[j, i]*v[t, j] for j in range(n_prods)))
            model.addConstr(v[t, i]<=d[t, i].item()-q[i]+M[i]*z[t, i])
            model.addConstr(v[t, i]>=d[t, i].item()-q[i]-M[i]*z[t, i])
            model.addConstr(v[t, i]<=d[t, i].item()*(1-z[t, i]))

    # solve and retrieve solution
    model.optimize()
    if model.status == GRB.OPTIMAL:
        orders = np.array([[q[p].X for p in range(n_prods)]])
    else:
        raise Exception('Optimization was not successful.')
    return orders, model.status

def solve_complex_parametric_seperate(y_train:np.array, y_train_pred:np.array, y_test_pred:np.array, scenario_size:int=100, n_threads:int=40):
    """Solve the complex newsvendor problem in a parametric way.

    Parameters
    ----------
    y_train : Demand data for training, shape (T, N_PRODUCTS)
    y_train_pred : Demand predictions for training, shape (T, N_PRODUCTS)
    y_test_pred : Demand predictions for testing, shape (T, N_PRODUCTS)
    u : Underage costs, shape (1, N_PRODUCTS)
    o : Overage costs, shape (1, N_PRODUCTS)
    alpha : Substitution rates, shape (N_PRODUCTS, N_PRODUCTS)
    scenario_size : Number of scenarios to sample for the approaches
    n_threads : Number of threads for the optimization

    Returns
    ----------
    final_order_quantities_parametric : np.array
        Final order quantities for each week in data_test, parametric approach
    """

    # load cost structure and get the number of products
    global alpha, underage, overage
    n_prods = y_train.shape[1] # number of products

    # Initialize an empty list to store the final order quantities
    final_order_quantities_parametric = []

    forecast_error = y_train - y_train_pred     #  (T,n)
    forecast_error_std = forecast_error.std(axis=0) # (n,)

    # Loop over each week in data_test
    for row in y_test_pred: # row (n,)

        # Initialize arrays of shape (scenario_size, n)
        demand_scenarios_parametric = np.zeros((scenario_size, n_prods))
        # Fill the arrays
        for i in range(scenario_size):
            demand_scenarios_parametric[i] = row + np.random.normal(loc=0, scale=forecast_error_std)

        # Initialize a list to store the solutions for each scenario
        saa_solutions_p = []        
        # For each demand scenario, solve the newsvendor problem
        for demand in demand_scenarios_parametric:
            # Calculate the solution for this scenario
            demand = demand.reshape(1,-1)
            solution, status = solve_MILP(d=demand, n_threads=n_threads)
            # Ensure solution is of shape (6,)
            solution = np.squeeze(solution)  # This changes shape from (1, 6) to (6,)
            # Store the solution for this scenario
            saa_solutions_p.append(solution)

        saa_solutions_p = np.array(saa_solutions_p)
        # Average the solutions to get the final allocation
        final_allocation_p = np.mean(saa_solutions_p, axis=0)

        # Store the final order quantities
        final_order_quantities_parametric.append(final_allocation_p)

    final_order_quantities_parametric = np.array(final_order_quantities_parametric)
    return final_order_quantities_parametric

def solve_complex_non_parametric_seperate(y_train:np.array, y_train_pred:np.array, y_test_pred:np.array, scenario_size:int=100, n_threads:int=40):
    """Solve the complex newsvendor problem in a non-parametric way.

    Parameters
    ----------
    y_train : Demand data for training, shape (T, N_PRODUCTS)
    y_train_pred : Demand predictions for training, shape (T, N_PRODUCTS)
    y_test_pred : Demand predictions for testing, shape (T, N_PRODUCTS)
    u : Underage costs, shape (1, N_PRODUCTS)
    o : Overage costs, shape (1, N_PRODUCTS)
    alpha : Substitution rates, shape (N_PRODUCTS, N_PRODUCTS)
    scenario_size : Number of scenarios to sample for the approaches
    n_threads : Number of threads for the optimization

    Returns
    ----------
    final_order_quantities_non_parametric : np.array
        Final order quantities for each week in data_test, non-parametric approach
    """
    global alpha, underage, overage
    n_prods = y_train.shape[1] # number of products

    # Initialize an empty list to store the final order quantities
    final_order_quantities_non_parametric = []

    # Calculate the forecast error
    forecast_error = y_train - y_train_pred     #  (T,n)

    # Loop over each week in data_test
    for row in y_test_pred: # row (n,)

        # Initialize arrays of shape (scenario_size, n)
        demand_scenarios_non_parametric = np.zeros((scenario_size, n_prods))
        # Fill the arrays
        for i in range(scenario_size):
            random_row = forecast_error[np.random.randint(forecast_error.shape[0])]
            demand_scenarios_non_parametric[i] = row + random_row

        # Initialize a list to store the solutions for each scenario
        saa_solutions_np = []
        for demand in demand_scenarios_non_parametric:
            # Calculate the solution for this scenario
            demand = demand.reshape(1,-1)
            solution, status = solve_MILP(d=demand, n_threads=n_threads)
            # Ensure solution is of shape (6,)
            solution = np.squeeze(solution)  # This changes shape from (1, 6) to (6,)
            # Store the solution for this scenario
            saa_solutions_np.append(solution)

        saa_solutions_np = np.array(saa_solutions_np)
        # Average the solutions to get the final allocation
        final_allocation_np = np.mean(saa_solutions_np, axis=0)
        # Store the final order quantities
        final_order_quantities_non_parametric.append(final_allocation_np)

    final_order_quantities_non_parametric = np.array(final_order_quantities_non_parametric)
    return final_order_quantities_non_parametric

def solve_basic_parametric_seperate(y_train:np.array, y_train_pred:np.array, y_test_pred:np.array, scenario_size:int=10, n_threads:int=40):
    """Solve the basic newsvendor problem in a parametric and a non-parametric way.

    Parameters
    ----------
    y_train : Demand data for training, shape (T, N_PRODUCTS)
    y_train_pred : Demand predictions for training, shape (T, N_PRODUCTS)
    y_test_pred : Demand predictions for testing, shape (T, N_PRODUCTS)
    u : Underage costs, shape (1, N_PRODUCTS)
    o : Overage costs, shape (1, N_PRODUCTS)
    scenario_size : Number of scenarios to sample for the approaches
    n_threads : Number of threads for the optimization

    Returns
    ----------
    final_order_quantities_parametric : np.array
        Final order quantities for each week in data_test, parametric approach
    """
    global alpha, underage, overage
    critical_ratio = underage / (underage + overage) # critical ratio
    
     # Initialize an empty list to store the final order quantities
    final_order_quantities_parametric = []

    y_train = y_train.reshape(-1,1) 
    y_train_pred = y_train_pred.reshape(-1,1)

    # Calculate the forecast error
    forecast_error = y_train - y_train_pred     #  (T,n)
    forecast_error_std = forecast_error.std(axis=0) # (n,)

    # Loop over each week in data_test
    for i in range(len(y_test_pred)):

        # Create Demand Scenarios for this week
        demand_scenarios_parametric = y_test_pred[i] + np.random.normal(loc=0, scale=forecast_error_std, size=scenario_size)

        # Initialize a list to store the solutions for each scenario
        saa_solutions_p = []

        # For each demand scenario, solve the newsvendor problem
        for demand in demand_scenarios_parametric:
            # Calculate the solution for this scenario
            solution = norm.ppf(critical_ratio,loc=demand,scale=forecast_error_std)
            # Store the solution for this scenario
            saa_solutions_p.append(solution)

        # Average the solutions to get the final allocation
        final_allocation_p = np.mean(saa_solutions_p, axis=0)

        # Store the final order quantities
        final_order_quantities_parametric.append(final_allocation_p)

    return final_order_quantities_parametric 

def solve_basic_non_parametric_seperate(y_train:np.array, y_train_pred:np.array, y_test_pred:np.array, scenario_size:int=10, n_threads:int=40):
    """Solve the basic newsvendor problem in a non-parametric way.

    Parameters
    ----------
    y_train : Demand data for training, shape (T, N_PRODUCTS)
    y_train_pred : Demand predictions for training, shape (T, N_PRODUCTS)
    y_test_pred : Demand predictions for testing, shape (T, N_PRODUCTS)
    u : Underage costs, shape (1, N_PRODUCTS)
    o : Overage costs, shape (1, N_PRODUCTS)
    scenario_size : Number of scenarios to sample for the approaches
    n_threads : Number of threads for the optimization

    Returns
    ----------
    final_order_quantities_non_parametric : np.array
        Final order quantities for each week in data_test, non-parametric approach
    """
    global alpha, underage, overage
    critical_ratio = underage / (underage + overage) # critical ratio
    
     # Initialize an empty list to store the final order quantities
    final_order_quantities_non_parametric = []

    y_train = y_train.reshape(-1,1) 
    y_train_pred = y_train_pred.reshape(-1,1)

    # Calculate the forecast error
    forecast_error = y_train - y_train_pred     #  (T,n)
    forecast_error_std = forecast_error.std(axis=0) # (n,)
  
    # Flatten the forecast_error
    forecast_error_flattened = np.ravel(forecast_error)

    # Loop over each week in data_test
    for i in range(len(y_test_pred)):

        # Create Demand Scenarios based on historical errors for this week (size determined by scenario_size)
        demand_scenarios_non_parametric = y_test_pred[i] + np.random.choice(forecast_error_flattened, size=scenario_size) 

        # Initialize a list to store the solutions for each scenario
        saa_solutions_np = []

        # For each demand scenario, solve the newsvendor problem
        for demand in demand_scenarios_non_parametric:
            # Calculate the solution for this scenario
            solution = norm.ppf(critical_ratio,loc=demand,scale=forecast_error_std)
            # Store the solution for this scenario
            saa_solutions_np.append(solution)

        # Average the solutions to get the final allocation
        final_allocation_np = np.mean(saa_solutions_np, axis=0)

        # Store the final order quantities
        final_order_quantities_non_parametric.append(final_allocation_np)

    return final_order_quantities_non_parametric

############################### ETS Functions ###########################################################################
        
def ets_forecast(y_train:np.array, y_val:np.array, y_test_length:int, verbose:int=0, fit_past:int = 12*7):
    """Forecast the demand using the ETS model

    Parameters
    ----------
    y_train : Demand data for training, shape (T, N_PRODUCTS)
    y_val : Demand data for validation, shape (T, N_PRODUCTS)
    y_test_length : Number of samples in the test set
    verbose : Verbosity level
    fit_past : Number of samples in the demand distribution estimate

    Returns
    ----------
    results_dct : dict
        Dictionary containing the results for each product
    """
    N_PRODUCTS = y_train.shape[1] # number of products
    N_TRAIN = y_train.shape[0] # number of training samples
    N_VAL = y_val.shape[0] # number of validation samples
    N_TEST = y_test_length # number of test samples

    # build all possible model configs
    error_types = ['add', 'mul']
    trend_types = ['add', 'mul', None]
    damped_trend_types = [True, False]
    seasonal_types = ['add', 'mul', None]
    seasonal_periods = [None, 7, 12, 24, 31]  
    model_configs = list(itertools.product(error_types, trend_types, damped_trend_types, seasonal_types, seasonal_periods))

    fit_past = fit_past # training data size (and number of samples in demand distribution estimate)
    timesteps = 1 # number of predictions timesteps (for our application, one-day-ahead predictions)
    
    # initialize variables to store the best configuration and its results
    best_config, best_rmse, best_mape, best_message = None, np.inf, np.inf, None
    best_rmse = np.inf
    results_dct = {}

    # loop over products
    for i in range(N_PRODUCTS):
        start_i = datetime.datetime.now() # computation start for product i
        results_dct[i] = {} # initialize result dict for product i
        target = np.append(y_train[:,i], y_val[:, i]) # train and val targets for i
        
        # loop through configurations
        for index, config in enumerate(model_configs, 1):
            preds = np.array([]) # variable in which validation set predictions are stored
            try:
                # loop through validation timesteps, fit and predict on-day-ahead
                for t in range(0, N_VAL, timesteps):
                    model = ETSModel(target[N_TRAIN+t-fit_past:N_TRAIN+t], *config)
                    model = model.fit()
                    preds = np.append(preds, np.maximum(0, model.forecast(timesteps)))

                # after evaluation is completed, compute RMSE and MAPE on validation set
                target_val = target[N_TRAIN:N_TRAIN+N_VAL]
                # Ensure preds and target_val have the same shape
                assert preds.shape == target_val.shape, "preds and target_val must have the same shape"

                rmse = np.sqrt( np.mean( (preds-target_val)**2 ) )
                mape = np.mean( abs(preds[target_val>0]-target_val[target_val>0]) / target_val[target_val>0] )

                message = 'success'
            except Exception as e:
                rmse, mape = np.inf, np.inf
                message = e
                print('Exeption \n', e)

            # If the RMSE is lower than the best RMSE so far, update the best configuration and its results
            if rmse < best_rmse:
                best_config = config
                best_rmse = rmse
                best_mape = mape
                best_message = message

        # fit the best model on the whole training set and predict for the test set
        model = ETSModel(target, *best_config)
        model = model.fit()
        best_test_pred = model.forecast(N_TEST)
        
        results_dct[i] = (best_config, best_rmse, best_mape, best_test_pred, best_message) # save results for product "i" and configuration "config

    return results_dct

def ets_evaluate(y_test:np.array, results_dct:dict):
    """Evaluate the ETS model

    Parameters
    ----------
    y_test : Demand data for testing, shape (T, N_PRODUCTS)
    results_dct : Dictionary containing the results for each product
    underage : Underage costs, shape (1, N_PRODUCTS)

    Returns
    ----------
    profit_ets_single : float
        Profit for the single product case
    profit_ets_multi : float
        Profit for the multi-product case
    """
    global alpha, underage, overage
    N_PRODUCTS = y_test.shape[1] # number of products
    N_TEST = y_test.shape[0] # number of test samples

    test_pred = np.zeros((N_TEST, N_PRODUCTS))  # Assuming N_TEST is the number of test samples
    
    for i in range(N_PRODUCTS):
        test_pred[:, i] = results_dct[i][3]

    y_test_single = y_test[:,0].reshape(-1, 1)
    test_pred_single = test_pred[:,0].reshape(-1, 1)

    profit_ets_single = np.mean(nvps_profit(y_test_single, test_pred_single ))
    profit_ets_multi = np.mean(nvps_profit(y_test, test_pred))

    return  profit_ets_single, profit_ets_multi

######################### Neural Network Functions ######################################################################

def make_nvps_loss():
    """ Create a custom loss function for the newsvendor problem under substitution
    
    Returns
    ---------
    nvps_loss : function
        Custom loss function
    """
    global alpha, underage, overage

    # transofrm the alpha, u, o to tensors
    u = tf.convert_to_tensor(underage, dtype=tf.float32) #underage costs
    o = tf.convert_to_tensor(overage, dtype=tf.float32) #overage costs
    a = tf.convert_to_tensor(alpha, dtype=tf.float32) #substitution matrix

    # define the loss function
    @tf.autograph.experimental.do_not_convert
    def nvps_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        q = tf.maximum(y_pred, 0.)

        # Calculate the demand increase for each product due to substitutions from other products
        demand_increase = tf.matmul( tf.maximum(0.0, y_true - y_pred),a)
        # Adjusted demand is the original demand plus the increase due to substitutions
        adjusted_demand = y_true + demand_increase
        # Calculate the profits
        profits = tf.matmul(q,u) - tf.matmul(tf.maximum(0.0,q - adjusted_demand), u+o)

        return -tf.math.reduce_mean(profits)
        
    return nvps_loss

def make_nvp_loss():
    """ Create a custom loss function for the newsvendor problem without substitution

    Returns
    ---------
    nvp_loss : function
        Custom loss function
    """
    global underage, overage
    q = underage / (underage + overage)
    @tf.autograph.experimental.do_not_convert
    def nvp_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        error = y_true - y_pred
        return tf.keras.backend.mean(tf.maximum(q*error, (q-1)*error), axis=-1)
    return nvp_loss

def create_NN_model(n_hidden:int, n_neurons:int, activation:str, input_shape:int, learning_rate:float, custom_loss, output_shape:int): 
    """ Build a neural network model with the specified architecture and hyperparameters    
    
    Parameters
    --------
    n_hidden : number of hidden layers
    n_neurons : number of neurons per hidden layer
    activation : activation function
    input_shape : number of features
    learning_rate : learning rate
    custom_loss : custom loss function
    output_shape : number of products

    Returns
    --------
    model : keras model
        Neural network model
    """
    # define the model
    model = Sequential()
    model.add(Input(shape=(input_shape,)))
    for layer in range(n_hidden):
        model.add(Dense(n_neurons, activation=activation))
    model.add(Dense(output_shape))
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss=custom_loss, optimizer=optimizer, metrics=None)
    return model

def tune_NN_model_optuna(X_train:np.array, y_train:np.array, X_val:np.array, y_val:np.array, integrated:bool, patience:int=10, 
                         verbose:int=0, trials:int=100, seed:int=42, threads:int=40): 
    """
        file_path:str, input_shape:int, output_shape:int, val_size:int, integrated:bool, patience:int=10, 
                         verbose:int=0, trials:int=100, seed:int=42, threads:int=40):
     Tune a neural network model on the given training data with early stopping using Optuna.
    
    Parameters
    --------------
    X_train : training feature data (samples, features)
    y_train : training targets (samples, N_PRODUCTS)
    X_val : validation feature data (samples, features)
    y_val : validation targets (samples, N_PRODUCTS)
    alpha_input : np.array
    Substitution rates, shape (N_PRODUCTS, N_PRODUCTS)
    patience : number of epochs without improvement before stopping
    verbose : keras' verbose parameter for silent / verbose model training
    trials : number of trials for the hyperparameter optimization

    Returns
    ----------
    best_estimator : keras model
        Best model found by Optuna
    hyperparameter : list
        Best hyperparameters found by Optuna
    study.best_value : float
        Best profit found by Optuna
    """
    global alpha, underage, overage
    input_shape = X_train.shape[1]
    output_shape = y_train.shape[1]

    # set seed for reproducability
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # Optuna hyperparameter optimization
    def objective(trial):

        # define the early stopping callback
        pruning_callback = KerasPruningCallback(trial, 'val_loss')

        # define the hyperparameters space
        n_hidden = trial.suggest_int('n_hidden', 0, 15) # changed from 10 to 15
        n_neurons = trial.suggest_int('n_neurons', 1, 30)
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1, log=True) # changed from 1e-1 to 1
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        epochs = trial.suggest_int('epochs', 10, 50)
        activation = trial.suggest_categorical('activation', ['relu', 'sigmoid', 'tanh'])

        # create a neural network model with basic hyperparameters
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience)

        # construct loss function based on the number of products
        if not integrated:
            custom_loss="mean_squared_error"
            model_ANN = KerasRegressor(model=create_NN_model, n_hidden=n_hidden, n_neurons=n_neurons, activation=activation,
                                input_shape=input_shape, learning_rate=learning_rate, custom_loss=custom_loss, output_shape=output_shape,
                                callbacks=[early_stopping])
        elif output_shape == 1 and integrated:
            custom_loss = make_nvp_loss() 
            model_ANN = KerasRegressor(model=create_NN_model, n_hidden=n_hidden, n_neurons=n_neurons, activation=activation,
                                input_shape=input_shape, learning_rate=learning_rate, custom_loss=custom_loss, output_shape=output_shape,
                                callbacks=[early_stopping])
        elif output_shape != 1 and integrated: 
            custom_loss = make_nvps_loss()
            model_ANN = KerasRegressor(model=create_NN_model, n_hidden=n_hidden, n_neurons=n_neurons, activation=activation,
                                input_shape=input_shape, learning_rate=learning_rate, custom_loss=custom_loss, output_shape=output_shape,
                                callbacks=[early_stopping])
        else:
            raise ValueError('Invalid Configuration')
        
        # Create the generators
        #train_generator = HDF5DataGenerator(file_path, 'X_train', 'y_train', batch_size=batch_size)
        #val_generator = HDF5DataGenerator(file_path, 'X_val', 'y_val', batch_size=batch_size)
        
        #pruning_callback = KerasPruningCallback(trial, 'val_loss')
        model_ANN.fit(X_train, y_train, validation_data=(X_val, y_val),  epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=[pruning_callback])
        
        # Create another generator for prediction
        #pred_generator = HDF5DataGenerator(file_path, 'X_val', 'y_val', batch_size=batch_size)

        # Make predictions on validation set and compute profits
        #q_val = model_ANN.predict_generator(pred_generator, steps=val_size//batch_size)
        q_val = model_ANN.predict(X_val)

        # If integrated, we can use the profit function, 
        #       otherwise we use the negative absolute error (otherwise we would "cheat")
        if integrated:
            result = np.mean(nvps_profit(y_val, q_val))
        else:
            result = -np.abs(np.mean(q_val-y_val))

        return result

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=trials, n_jobs=threads, gc_after_trial=True)


    # Get the best parameters and best estimator
    best_params = study.best_params
    hyperparameter = [best_params['n_hidden'], best_params['n_neurons'],best_params['learning_rate'], 
                    best_params['epochs'], patience, best_params['batch_size'], best_params['activation']]
    best_estimator = train_NN_model(hp=hyperparameter,X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, integrated=integrated,verbose=verbose)
    
    return best_estimator, hyperparameter, study.best_value

def train_NN_model(hp:list, X_train, y_train, X_val, y_val, integrated:bool, verbose:int=0):
    """ file_path:str, input_shape:int, output_shape:int,
    
    Train a network on the given training data with early stopping.
    
    Parameters
    --------------
    hp : hyperparameters in the following order: hidden_nodes, lr, max_epochs, patience, batch_size
    X_train : training feature data
    y_train : training targets
    X_val : validation feature data
    y_val : validation targets
    verbose : keras' verbose parameter for silent / verbose model training

    Returns
    ----------
    model : keras model
        Final model
    """
    # set seed for reproducability
    tf.random.set_seed(42)
    np.random.seed(42)

    global alpha, underage, overage
    input_shape = X_train.shape[1]
    output_shape = y_train.shape[1]

    # construct loss function based on the number of products
    if  integrated == False:
        loss = tf.keras.losses.MeanSquaredError()
    elif (output_shape == 1) & ( integrated == True):
        loss =  make_nvp_loss()
    elif (output_shape != 1) & ( integrated == True): 
        loss =  make_nvps_loss()

    # extract hyperparameters, build and compile MLP
    hidden_nodes, n_neurons, lr, max_epochs, patience, batch_size, activation = hp
    mlp =  create_NN_model(n_hidden=hidden_nodes, n_neurons=n_neurons, activation=activation, 
                        input_shape=input_shape, learning_rate=lr, custom_loss=loss, 
                        output_shape=output_shape)

    # train MLP with early stopping
    callback = EarlyStopping(monitor='val_loss', patience=patience)

    # Create the generators
    #train_generator = HDF5DataGenerator(file_path, 'X_train', 'y_train', batch_size=batch_size)
    #val_generator = HDF5DataGenerator(file_path, 'X_val', 'y_val', batch_size=batch_size)

    mlp.fit(X_train, y_train, epochs=max_epochs, batch_size=batch_size, validation_data=(X_val, y_val),
            verbose=verbose, callbacks=[callback])
    
    return mlp

######################### XGBoost Functions ######################################################################

def gradient(predt: np.ndarray, dtrain: xgb.DMatrix, ) -> np.ndarray:
    """ Calculate the gradient of the custom loss function"""
    global alpha, underage, overage
    try:
        y = dtrain.get_label().reshape(predt.shape)
        d = y + np.matmul(np.maximum(0, y - predt), alpha)
        u = underage.T
        o = overage.T
        return (-(u * np.maximum(0,d-predt) - o * np.maximum(0, predt-d))).reshape(y.size)
    except:
        logger.error(f"Error calculating gradient")
        raise
                
def hessian(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
    """ Calculate the hessian of the custom loss function"""
    return np.ones(predt.shape).reshape(predt.size)
        
def custom_loss(predt: np.ndarray, dtrain: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """ Calculate the gradient and hessian of the custom loss function"""
    grad =  gradient(predt, dtrain)
    hess =  hessian(predt, dtrain)
    return grad, hess

def tune_XGB_model(X_train:np.array, y_train:np.array, X_val:np.array, y_val:np.array, 
                   integrated:bool, patience:int=10, verbose:int=0, trials:int=100, threads:int=40):
    """ Tune a XGBoost model on the given training data with early stopping using Optuna.
    
    Parameters
    --------------
    X_train : training feature data (samples, features)
    y_train : training targets (samples, N_PRODUCTS)
    X_val : validation feature data (samples, features)
    y_val : validation targets (samples, N_PRODUCTS)
    patience : number of epochs without improvement before stopping
    verbose : verbosity level
    trials : number of trials for the optimization
    
    Returns
    ----------
    final_booster : xgb.Booster
        Final model
    best_params : dict
        Best hyperparameters found by Optuna
    results : dict
        Results of the optimization
    """
    global alpha, underage, overage

    if y_train.shape[1] == 1 and integrated == True:
        multi_strategy = "one_output_per_tree"
        custom_objective = "reg:quantileerror"
        quantile = underage / (underage + overage)
    elif y_train.shape[1] != 1 and integrated == True:
        multi_strategy = "multi_output_tree"
        custom_objective = custom_loss
        quantile = 0
    elif y_train.shape[1] == 1 and integrated == False:
        multi_strategy = "one_output_per_tree"
        custom_objective = 'reg:squarederror'
        quantile = 0
    elif y_train.shape[1] != 1 and integrated == False:
        multi_strategy = "multi_output_tree"
        custom_objective = "reg:squarederror"
        quantile = 0
    else:
        raise ValueError('Invalid Configuration')
  
    results = {} # initialize results dict
    
    # Optuna hyperparameter optimization
    def objective(trial):

        # Transform training and validation data to DMatrix
        X, y = X_train, y_train  
        Xy = xgb.DMatrix(X, label=y)
        dval = xgb.DMatrix(X_val, label=y_val)

        # if custom objective is used, we need to pass the custom loss function
        if custom_objective != custom_loss:
            params = {
                "tree_method": "hist",
                "num_target": y.shape[1],
                "multi_strategy": multi_strategy,
                "learning_rate": trial.suggest_float("learning_rate", 0.1, 1), # changed from 0.5 to 1
                "max_depth": trial.suggest_int("max_depth", 2, 9),  # changed from 6 to 9
                "subsample": trial.suggest_float("subsample", 0.3, 1), # changed from 0.9 to 1
                "quantile_alpha": quantile,
                "objective": custom_objective,
            }
            booster = xgb.train(
                params,
                dtrain=Xy,
                num_boost_round=128,
                evals=[(dval, "val")],
                evals_result=results,
                early_stopping_rounds=patience,
                verbose_eval=verbose
                         
            )
        # if no custom objective is used, we can use the default objective
        else:
            params = {
                "tree_method": "hist",
                "num_target": y.shape[1],
                "multi_strategy": multi_strategy,
                "learning_rate": trial.suggest_float("learning_rate", 0.1, 0.5),
                "max_depth": trial.suggest_int("max_depth", 2, 6),
                "subsample": trial.suggest_float("subsample", 0.3, 0.9),
                "quantile_alpha": quantile,
            }
            booster = xgb.train(
            params,
                dtrain=Xy,
                num_boost_round=128,
                obj=custom_objective,
                evals=[(dval, "val")],
                evals_result=results,
                early_stopping_rounds=patience,
                verbose_eval=verbose
            )
    
        # make predictions on validation set and compute profits
        val_set = xgb.DMatrix(X_val)
        q_val = booster.predict(val_set)

        # If integrated, we can use the profit function, 
        #       otherwise we use the negative absolute error (otherwise we would "cheat")
        if integrated:
            result = np.mean(nvps_profit(y_val, q_val))
        else:
            result = -np.abs(np.mean(q_val-y_val))
        return result

    # Create the study and optimize the objective function
    study = optuna.create_study(direction='maximize')
    
    study.optimize(objective, n_trials=trials, n_jobs=threads, gc_after_trial=True)

    # Transform training and validation data to DMatrix
    X, y = X_train, y_train  
    Xy = xgb.DMatrix(X, label=y)
    dval = xgb.DMatrix(X_val, label=y_val)

    # Get the best parameters
    best_params = study.best_trial.params

    # Add the fixed parameters
    if custom_objective !=  custom_loss:
        best_params.update({
            "tree_method": "hist",
            "num_target": y.shape[1],
            "multi_strategy": multi_strategy,
            "quantile_alpha": quantile,
            "objective": custom_objective,
        })
    else:
        best_params.update({
            "tree_method": "hist",
            "num_target": y.shape[1],
            "multi_strategy": multi_strategy,
        })

    # Train the final model
    if custom_objective != custom_loss:
        final_booster = xgb.train(
            best_params,
            dtrain=Xy,
            num_boost_round=128,
            evals=[(dval, "val")],
            verbose_eval=verbose
        )
    else:
        final_booster = xgb.train(
            best_params,
            dtrain=Xy,
            num_boost_round=128,
            obj=custom_objective,
            evals=[(dval, "val")],
            verbose_eval=verbose
        )
    # return the final model, best parameters and results
    return final_booster, best_params, results

def train_XGB_model(hyperparameter:dict, X_train:np.array, y_train:np.array, X_val:np.array, y_val:np.array):
    """ Train a XGBoost model on the given training data with given hyperparameters.

    Parameters
    --------------
    hyperparameter : hyperparameters for the XGBoost model
    X_train : training feature data
    y_train : training targets
    X_val : validation feature data
    y_val : validation targets

    Returns
    ----------
    final_booster : xgb.Booster
        Final model
    """
    # Train the final model
    Xy = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    final_booster = xgb.train(
        hyperparameter,
        dtrain=Xy,
        num_boost_round=128,
        obj= custom_loss,
        evals=[(dval, "val")]
    )
    return final_booster

############################################################### Approach Handler ########################################################

def ioa_ann_simple(X_train:np.array, y_train:np.array, X_val:np.array, y_val:np.array,  X_test:np.array, y_test:np.array, 
                   underage_data_single:np.array, overage_data_single:np.array, trials:int, dataset_id:str, path:str):
    """ Train and evaluate the integrated optimization approach with a simple ANN model and saves model, hyperparameters and profit

    Parameters
    ----------
    X_train : training feature data
    y_train : training targets
    X_val : validation feature data
    y_val : validation targets
    X_test : testing feature data
    y_test : testing targets
    underage_data_single : underage costs
    overage_data_single : overage costs
    dataset_id : dataset identifier
    path : path to save the model
    """
    # Initialize for measurement of memory usage and elapsed time
    monitor = MemoryMonitor(interval=1)  # monitor every second
    monitor.start()
    start = datetime.datetime.now()

    # Integrated Optimization Approach - ANN - simple:
    load_cost_structure(alpha_input=None, underage_input=underage_data_single, overage_input=overage_data_single) # Initialize the cost structure
    # Tune the ANN model with Optuna
    model_ANN_simple, hyperparameter, val_profit = tune_NN_model_optuna(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, integrated=True, trials=trials)
    # Make predictions on the test set
    target_prediction_ANN = model_ANN_simple.predict(X_test)
    # Calculate the profit of the predictions
    profit_simple_ANN_IOA = nvps_profit(demand=y_test, q=target_prediction_ANN)

    # Measure memory usage and elapsed time
    monitor.stop() # stop the average memory monitor
    avg_memory = monitor.average_memory_usage() # read the average memory monitor
    peak_memory = monitor.peak_memory_usage() # read the peak memory monitor
    end = datetime.datetime.now() # stop the time monitor
    elapsed = (end-start).total_seconds()

    # Save the model, hyperparameters, profit, time and memory usage
    save_model(model=model_ANN_simple, hyperparameter=hyperparameter, profit=profit_simple_ANN_IOA, elapsed=elapsed, peak_memory=peak_memory, avg_memory=avg_memory, dataset_id=dataset_id, path=path, name='ANN_simple_IOA')



def soa_ann_simple(X_train:np.array, y_train:np.array, X_val:np.array, y_val:np.array, X_test:np.array, y_test:np.array, 
                   underage_data_single:np.array, overage_data_single:np.array, trials:int, dataset_id:str, path:str):
    """ Train and evaluate the seperate optimization approach with a simple ANN model and saves model, hyperparameters and profit

    Parameters
    ----------
    X_train : training feature data
    y_train : training targets
    X_val : validation feature data
    y_val : validation targets
    X_test : testing feature data
    y_test : testing targets
    underage_data_single : underage costs
    overage_data_single : overage costs
    dataset_id : dataset identifier
    path : path to save the model
    """
    # Initialize for measurement of memory usage and elapsed time
    monitor_1 = MemoryMonitor(interval=1)  # monitor every second
    monitor_2 = MemoryMonitor(interval=1)  # monitor every second
    monitor_3 = MemoryMonitor(interval=1)  # monitor every second
    monitor_1.start() # start average memory monitor for estimation
    start = datetime.datetime.now() # start the time monitor for estimation

    # Seperate Optimization Approach - ANN - simple:
    load_cost_structure(alpha_input=None, underage_input=underage_data_single, overage_input=overage_data_single) # Initialize the cost structure
    # Tune the ANN model with Optuna
    model_ANN_simple, hyperparameter, val_profit = tune_NN_model_optuna(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, integrated=False, trials=trials)
    # Make predictions on the test and train set
    target_prediction_ANN = model_ANN_simple.predict(X_test)
    train_prediction_ANN = model_ANN_simple.predict(X_train)

    # Measure memory usage and elapsed time for estimation
    monitor_1.stop() # stop the average memory monitor for estimation
    checkpoint_1 = datetime.datetime.now() # checkpoint for the time monitor
    monitor_2.start() # start the average memory monitor for the parametric optimization

    # Calculate the orders and profits in a parametric way
    orders_ssp_ann = solve_basic_parametric_seperate(y_train=y_train, y_train_pred=train_prediction_ANN, y_test_pred=target_prediction_ANN)
    profit_ssp_ANN = nvps_profit(demand=y_test, q=orders_ssp_ann)

    # Measure memory usage and elapsed time for parametric optimization
    monitor_2.stop() # stop the average memory monitor for the parametric optimization
    checkpoint_2 = datetime.datetime.now() # checkpoint for the time monitor
    monitor_3.start() # start the average memory monitor for the non-parametric optimization 

    # Calculate the orders and profits in a non-parametric way
    orders_ssnp_ann = solve_basic_non_parametric_seperate(y_train=y_train, y_train_pred=train_prediction_ANN, y_test_pred=target_prediction_ANN)
    profit_ssnp_ANN = nvps_profit(demand=y_test, q=orders_ssnp_ann)

    # Measure memory usage and elapsed time for non-parametric optimization
    monitor_3.stop() # stop the average memory monitor for the non-parametric optimization
    end = datetime.datetime.now() # stop the time monitor 

    # Calculate the parametric and non-parametric metadata
    peak_memory_ssp = np.maximum(monitor_2.peak_memory_usage(), monitor_1.peak_memory_usage())
    peak_memory_ssnp = np.maximum(monitor_3.peak_memory_usage(), monitor_1.peak_memory_usage())
    elapsed_ssp = (checkpoint_2-start).total_seconds()
    elapsed_ssnp = ((checkpoint_1-start)+(end-checkpoint_2)).total_seconds()
    avg_memory_ssp = np.maximum(monitor_1.average_memory_usage(), monitor_2.average_memory_usage())
    avg_memory_ssnp = np.maximum(monitor_1.average_memory_usage(), monitor_3.average_memory_usage())

    # Save the model, hyperparameters, profit, time and memory usage
    save_model(model=model_ANN_simple, hyperparameter=hyperparameter, profit=profit_ssp_ANN, elapsed=elapsed_ssp, peak_memory=peak_memory_ssp, avg_memory=avg_memory_ssp, dataset_id=dataset_id, path=path, name='ANN_simple_SOAp')
    save_model(model=model_ANN_simple, hyperparameter=hyperparameter, profit=profit_ssnp_ANN, elapsed=elapsed_ssnp, peak_memory=peak_memory_ssnp, avg_memory=avg_memory_ssnp, dataset_id=dataset_id, path=path, name='ANN_simple_SOAnp')

def ioa_xgb_simple(X_train:np.array, y_train:np.array, X_val:np.array, y_val:np.array, X_test:np.array, y_test:np.array, 
                   underage_data_single:np.array, overage_data_single:np.array, trials:int, dataset_id:str, path:str):
    """ Train and evaluate the integrated optimization approach with a simple XGBoost model and saves model, hyperparameters and profit

    Parameters
    ----------
    X_train : training feature data
    y_train : training targets
    X_val : validation feature data
    y_val : validation targets
    X_test : testing feature data
    y_test : testing targets
    underage_data_single : underage costs
    overage_data_single : overage costs
    dataset_id : dataset identifier
    path : path to save the model
    """
    # Initialize for measurement of memory usage and elapsed time
    monitor = MemoryMonitor(interval=1)  # monitor every second
    monitor.start()
    start = datetime.datetime.now()

    # Integrated Optimization Approach - XGBoost - Simple:
    load_cost_structure(alpha_input=None, underage_input=underage_data_single, overage_input=overage_data_single) # Initialize the cost structure
    # Tune the XGBoost model with Optuna
    xgb_model, params, results = tune_XGB_model(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, integrated=True, trials=trials)
    # Make predictions on the test set  
    xgb_result = xgb_model.predict(xgb.DMatrix(X_test))
    # Calculate the profit of the predictions
    profit_simple_XGB_IOA = nvps_profit(demand=y_test, q=xgb_result)

    # Measure memory usage and elapsed time
    monitor.stop() # stop the average memory monitor
    avg_memory = monitor.average_memory_usage() # read the average memory monitor
    peak_memory = monitor.peak_memory_usage() # read the peak memory monitor
    end = datetime.datetime.now() # stop the time monitor
    elapsed = (end-start).total_seconds()
 
    # Save the model, hyperparameters, profit, time and memory usage
    save_model(model=xgb_model, hyperparameter=params, profit=profit_simple_XGB_IOA, elapsed=elapsed, peak_memory=peak_memory, avg_memory=avg_memory, dataset_id=dataset_id, path=path, name='XGB_simple_IOA')

def soa_xgb_simple(X_train:np.array, y_train:np.array, X_val:np.array, y_val:np.array, X_test:np.array, y_test:np.array, 
                   underage_data_single:np.array, overage_data_single:np.array, trials:int, dataset_id:str, path:str):
    """ Train and evaluate the seperate optimization approach with a simple XGBoost model and saves model, hyperparameters and profit

    Parameters
    ----------
    X_train : training feature data
    y_train : training targets
    X_val : validation feature data
    y_val : validation targets
    X_test : testing feature data
    y_test : testing targets
    underage_data_single : underage costs
    overage_data_single : overage costs
    dataset_id : dataset identifier
    path : path to save the model
    """
    # Initialize for measurement of memory usage and elapsed time
    monitor_1 = MemoryMonitor(interval=1)  # monitor every second
    monitor_2 = MemoryMonitor(interval=1)  # monitor every second
    monitor_3 = MemoryMonitor(interval=1)  # monitor every second
    monitor_1.start() # start average memory monitor for estimation
    start = datetime.datetime.now() # start the time monitor for estimation

    # Seperated Optimization Approach - XGBoost - Simple:
    load_cost_structure(alpha_input=None, underage_input=underage_data_single, overage_input=overage_data_single) # Initialize the cost structure
    xgb_model, hyperparameter_XGB_SOA_Complex, val_profit = tune_XGB_model(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, integrated=False, trials=trials)
    # Make predictions on the test and train set
    target_prediction_XGB = xgb_model.predict(xgb.DMatrix(X_test))
    train_prediction_XGB = xgb_model.predict(xgb.DMatrix(X_train))  

    # Measure memory usage and elapsed time for estimation
    monitor_1.stop() # stop the average memory monitor for estimation
    checkpoint_1 = datetime.datetime.now() # checkpoint for the time monitor
    monitor_2.start() # start the average memory monitor for the parametric optimization

    # Calculate the orders and profits in a parametric way
    orders_ssp_xgb = solve_basic_parametric_seperate(y_train=y_train, y_train_pred=train_prediction_XGB, y_test_pred=target_prediction_XGB)
    profit_ssp_XGB = nvps_profit(demand=y_test, q=orders_ssp_xgb)

    # Measure memory usage and elapsed time for parametric optimization
    monitor_2.stop() # stop the average memory monitor for the parametric optimization
    checkpoint_2 = datetime.datetime.now() # checkpoint for the time monitor
    monitor_3.start() # start the average memory monitor for the non-parametric optimization 

    # Calculate the orders and profits in a non-parametric way
    orders_ssnp_xgb = solve_basic_non_parametric_seperate(y_train=y_train, y_train_pred=train_prediction_XGB, y_test_pred=target_prediction_XGB)
    profit_ssnp_XGB = nvps_profit(demand=y_test, q=orders_ssnp_xgb)

    # Measure memory usage and elapsed time for non-parametric optimization
    monitor_3.stop() # stop the average memory monitor for the non-parametric optimization
    end = datetime.datetime.now() # stop the time monitor 

    # Calculate the parametric and non-parametric metadata
    peak_memory_ssp = np.maximum(monitor_2.peak_memory_usage(), monitor_1.peak_memory_usage())
    peak_memory_ssnp = np.maximum(monitor_3.peak_memory_usage(), monitor_1.peak_memory_usage())
    elapsed_ssp = (checkpoint_2-start).total_seconds()
    elapsed_ssnp = ((checkpoint_1-start)+(end-checkpoint_2)).total_seconds()
    avg_memory_ssp = np.maximum(monitor_1.average_memory_usage(), monitor_2.average_memory_usage())
    avg_memory_ssnp = np.maximum(monitor_1.average_memory_usage(), monitor_3.average_memory_usage())

    # Save the model, hyperparameters, profit, time and memory usage
    save_model(model=xgb_model, hyperparameter=hyperparameter_XGB_SOA_Complex, profit=profit_ssp_XGB, elapsed=elapsed_ssp, peak_memory=peak_memory_ssp, avg_memory=avg_memory_ssp, dataset_id=dataset_id, path=path, name='XGB_simple_SOAp')
    save_model(model=xgb_model, hyperparameter=hyperparameter_XGB_SOA_Complex, profit=profit_ssnp_XGB, elapsed=elapsed_ssnp, peak_memory=peak_memory_ssnp, avg_memory=avg_memory_ssnp, dataset_id=dataset_id, path=path, name='XGB_simple_SOAnp')

def ioa_ann_complex(X_train:np.array, y_train:np.array, X_val:np.array, y_val:np.array, X_test:np.array, y_test:np.array, 
                    alpha_data:np.array, underage_data:np.array, overage_data:np.array, trials:int, dataset_id:str, path:str):
    """ Train and evaluate the integrated optimization approach with a complex ANN model and saves model, hyperparameters and profit
    
    Parameters
    ----------
    X_train : training feature data
    y_train : training targets
    X_val : validation feature data
    y_val : validation targets
    X_test : testing feature data
    y_test : testing targets
    alpha_data : substitution rates
    underage_data : underage costs
    overage_data : overage costs
    dataset_id : dataset identifier
    path : path to save the model
    """
    # Initialize for measurement of memory usage and elapsed time
    monitor = MemoryMonitor(interval=1)  # monitor every second
    monitor.start()
    start = datetime.datetime.now()

    # Integrated Optimization Approach - ANN - complex:
    load_cost_structure(alpha_input=alpha_data, underage_input=underage_data, overage_input=overage_data) # Initialize the cost structure
    # Tune the ANN model with Optuna
    model_ANN_complex, hyperparameter, val_profit = tune_NN_model_optuna(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, integrated=True, trials=trials)
    # Make predictions on the test set
    target_prediction_ANN = model_ANN_complex.predict(X_test)
    # Calculate the profit of the predictions
    profit_complex_ANN_IOA = nvps_profit(demand=y_test, q=target_prediction_ANN)

    # Measure memory usage and elapsed time
    monitor.stop() # stop the average memory monitor
    avg_memory = monitor.average_memory_usage() # read the average memory monitor
    peak_memory = monitor.peak_memory_usage() # read the peak memory monitor
    end = datetime.datetime.now() # stop the time monitor
    elapsed = (end-start).total_seconds()

    # Save the model, hyperparameters, profit, time and memory usage
    save_model(model=model_ANN_complex, hyperparameter=hyperparameter, profit=profit_complex_ANN_IOA, elapsed=elapsed, peak_memory=peak_memory, avg_memory=avg_memory, dataset_id=dataset_id, path=path, name='ANN_complex_IOA')

def soa_ann_complex(X_train:np.array, y_train:np.array, X_val:np.array, y_val:np.array, X_test:np.array, y_test:np.array, 
                    alpha_data:np.array, underage_data:np.array, overage_data:np.array, trials:int, dataset_id:str, path:str):
    """ Train and evaluate the seperate optimization approach with a complex ANN model and saves model, hyperparameters and profit

    Parameters
    ----------
    X_train : training feature data
    y_train : training targets
    X_val : validation feature data
    y_val : validation targets
    X_test : testing feature data
    y_test : testing targets
    alpha_data : substitution rates
    underage_data : underage costs
    overage_data : overage costs
    dataset_id : dataset identifier
    path : path to save the model
    """
    # Initialize for measurement of memory usage and elapsed time
    monitor_1 = MemoryMonitor(interval=1)  # monitor every second
    monitor_2 = MemoryMonitor(interval=1)  # monitor every second
    monitor_3 = MemoryMonitor(interval=1)  # monitor every second
    monitor_1.start() # start average memory monitor for estimation
    start = datetime.datetime.now() # start the time monitor for estimation

    # Seperate Optimization Approach - ANN - complex:
    load_cost_structure(alpha_input=alpha_data, underage_input=underage_data, overage_input=overage_data) # Initialize the cost structure
    # Tune the ANN model with Optuna
    model_ANN_complex, hyperparameter, val_profit = tune_NN_model_optuna(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, integrated=False, trials=trials)
    # Make predictions on the test and train set
    target_prediction_ANN = model_ANN_complex.predict(X_test)
    train_prediction_ANN = model_ANN_complex.predict(X_train)

    # Measure memory usage and elapsed time for estimation
    monitor_1.stop() # stop the average memory monitor for estimation
    checkpoint_1 = datetime.datetime.now() # checkpoint for the time monitor
    monitor_2.start() # start the average memory monitor for the parametric optimization

    # Calculate the orders and profits in a parametric way
    orders_scp_ann = solve_complex_parametric_seperate(y_train=y_train, y_train_pred=train_prediction_ANN, y_test_pred=target_prediction_ANN)
    profit_scp_ANN = nvps_profit(demand=y_test, q=orders_scp_ann)

    # Measure memory usage and elapsed time for parametric optimization
    monitor_2.stop() # stop the average memory monitor for the parametric optimization
    checkpoint_2 = datetime.datetime.now() # checkpoint for the time monitor
    monitor_3.start() # start the average memory monitor for the non-parametric optimization 

    # Calculate the orders and profits in a non-parametric way
    orders_scnp_ann = solve_complex_non_parametric_seperate(y_train=y_train, y_train_pred=train_prediction_ANN, y_test_pred=target_prediction_ANN)
    profit_scnp_ANN = nvps_profit(demand=y_test, q=orders_scnp_ann)

    # Measure memory usage and elapsed time for non-parametric optimization
    monitor_3.stop() # stop the average memory monitor for the non-parametric optimization
    end = datetime.datetime.now() # stop the time monitor 

    # Calculate the parametric and non-parametric metadata
    peak_memory_scp = np.maximum(monitor_2.peak_memory_usage(), monitor_1.peak_memory_usage())
    peak_memory_scnp = np.maximum(monitor_3.peak_memory_usage(), monitor_1.peak_memory_usage())
    elapsed_scp = (checkpoint_2-start).total_seconds()
    elapsed_scnp = ((checkpoint_1-start)+(end-checkpoint_2)).total_seconds()
    avg_memory_scp = np.maximum(monitor_1.average_memory_usage(), monitor_2.average_memory_usage())
    avg_memory_scnp = np.maximum(monitor_1.average_memory_usage(), monitor_3.average_memory_usage())

    # Save the model, hyperparameters, profit, time and memory usage
    save_model(model=model_ANN_complex, hyperparameter=hyperparameter, profit=profit_scp_ANN, elapsed=elapsed_scp, peak_memory=peak_memory_scp, avg_memory=avg_memory_scp, dataset_id=dataset_id, path=path, name='ANN_complex_SOAp')
    save_model(model=model_ANN_complex, hyperparameter=hyperparameter, profit=profit_scnp_ANN, elapsed=elapsed_scnp, peak_memory=peak_memory_scnp, avg_memory=avg_memory_scnp, dataset_id=dataset_id, path=path, name='ANN_complex_SOAnp')

def ioa_xgb_complex(X_train:np.array, y_train:np.array, X_val:np.array, y_val:np.array, X_test:np.array, y_test:np.array, 
                    alpha_data:np.array, underage_data:np.array, overage_data:np.array, trials:int, dataset_id:str, path:str):
    """ Train and evaluate the integrated optimization approach with a complex XGBoost model and saves model, hyperparameters and profit
    
    Parameters
    ----------
    X_train : training feature data
    y_train : training targets
    X_val : validation feature data
    y_val : validation targets
    X_test : testing feature data
    y_test : testing targets
    alpha_data : substitution rates
    underage_data : underage costs
    overage_data : overage costs
    dataset_id : dataset identifier
    path : path to save the model
    """
    # Initialize for measurement of memory usage and elapsed time
    monitor = MemoryMonitor(interval=1)  # monitor every second
    monitor.start()
    start = datetime.datetime.now()

    # Integrated Optimization Approach - XGBoost - Complex:
    load_cost_structure(alpha_input=alpha_data, underage_input=underage_data, overage_input=overage_data) # Initialize the cost structure
    # Tune the XGBoost model with Optuna
    xgb_model, params, results = tune_XGB_model(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, integrated=True, trials=trials)
    # Make predictions on the test set
    xgb_result = xgb_model.predict(xgb.DMatrix(X_test))
    # Calculate the profit of the predictions
    profit_complex_XGB_IOA = nvps_profit(demand=y_test, q=xgb_result) 

    # Measure memory usage and elapsed time
    monitor.stop() # stop the average memory monitor
    avg_memory = monitor.average_memory_usage() # read the average memory monitor
    peak_memory = monitor.peak_memory_usage() # read the peak memory monitor
    end = datetime.datetime.now() # stop the time monitor
    elapsed = (end-start).total_seconds()

    # Save the model, hyperparameters, profit, time and memory usage
    save_model(model=xgb_model, hyperparameter=params, profit=profit_complex_XGB_IOA, elapsed=elapsed, peak_memory=peak_memory, avg_memory=avg_memory, dataset_id=dataset_id, path=path, name='XGB_complex_IOA')

def soa_xgb_complex(X_train:np.array, y_train:np.array, X_val:np.array, y_val:np.array, X_test:np.array, y_test:np.array, 
                    alpha_data:np.array, underage_data:np.array, overage_data:np.array, trials:int, dataset_id:str, path:str):
    """ Train and evaluate the seperate optimization approach with a complex XGBoost model and saves model, hyperparameters and profit
    
    Parameters
    ----------
    X_train : training feature data
    y_train : training targets
    X_val : validation feature data
    y_val : validation targets
    X_test : testing feature data
    y_test : testing targets
    alpha_data : substitution rates
    underage_data : underage costs
    overage_data : overage costs
    dataset_id : dataset identifier
    path : path to save the model
    """
    # Initialize for measurement of memory usage and elapsed time
    monitor_1 = MemoryMonitor(interval=1)  # monitor every second
    monitor_2 = MemoryMonitor(interval=1)  # monitor every second
    monitor_3 = MemoryMonitor(interval=1)  # monitor every second
    monitor_1.start() # start average memory monitor for estimation
    start = datetime.datetime.now() # start the time monitor for estimation

    # Seperated Optimization Approach - XGBoost - Complex:
    load_cost_structure(alpha_input=alpha_data, underage_input=underage_data, overage_input=overage_data) # Initialize the cost structure
    # Tune the XGBoost model with Optuna
    xgb_model, hyperparameter_XGB_SOA_Complex, val_profit = tune_XGB_model(X_train, y_train, X_val, y_val, integrated=False, trials=trials)
    # Make predictions on the test and train set
    target_prediction_XGB = xgb_model.predict(xgb.DMatrix(X_test))
    train_prediction_XGB = xgb_model.predict(xgb.DMatrix(X_train))

    # Measure memory usage and elapsed time for estimation
    monitor_1.stop() # stop the average memory monitor for estimation
    checkpoint_1 = datetime.datetime.now() # checkpoint for the time monitor
    monitor_2.start() # start the average memory monitor for the parametric optimization

    # Calculate the orders and profits in a parametric way
    orders_scp_xgb = solve_complex_parametric_seperate(y_train=y_train, y_train_pred=train_prediction_XGB, y_test_pred=target_prediction_XGB)
    profit_scp_XGB = nvps_profit(demand=y_test, q=orders_scp_xgb)

    # Measure memory usage and elapsed time for parametric optimization
    monitor_2.stop() # stop the average memory monitor for the parametric optimization
    checkpoint_2 = datetime.datetime.now() # checkpoint for the time monitor
    monitor_3.start() # start the average memory monitor for the non-parametric optimization 

    # Calculate the orders and profits in a non-parametric way
    orders_scnp_xgb = solve_complex_non_parametric_seperate(y_train=y_train, y_train_pred=train_prediction_XGB, y_test_pred=target_prediction_XGB)
    profit_scnp_XGB = nvps_profit(demand=y_test, q=orders_scnp_xgb)

    # Measure memory usage and elapsed time for non-parametric optimization
    monitor_3.stop() # stop the average memory monitor for the non-parametric optimization
    end = datetime.datetime.now() # stop the time monitor 

    # Calculate the parametric and non-parametric metadata
    peak_memory_scp = np.maximum(monitor_2.peak_memory_usage(), monitor_1.peak_memory_usage())
    peak_memory_scnp = np.maximum(monitor_3.peak_memory_usage(), monitor_1.peak_memory_usage())
    elapsed_scp = (checkpoint_2-start).total_seconds()
    elapsed_scnp = ((checkpoint_1-start)+(end-checkpoint_2)).total_seconds()
    avg_memory_scp = np.maximum(monitor_1.average_memory_usage(), monitor_2.average_memory_usage())
    avg_memory_scnp = np.maximum(monitor_1.average_memory_usage(), monitor_3.average_memory_usage())

    # Save the model, hyperparameters, profit, time and memory usage
    save_model(model=xgb_model, hyperparameter=hyperparameter_XGB_SOA_Complex, profit=profit_scp_XGB, elapsed=elapsed_scp, peak_memory=peak_memory_scp, avg_memory=avg_memory_scp, dataset_id=dataset_id, path=path, name='XGB_complex_SOAp')
    save_model(model=xgb_model, hyperparameter=hyperparameter_XGB_SOA_Complex, profit=profit_scnp_XGB, elapsed=elapsed_scnp, peak_memory=peak_memory_scnp, avg_memory=avg_memory_scnp, dataset_id=dataset_id, path=path, name='XGB_complex_SOAnp')

def ets_baseline(y_train, y_val, y_test, underage_data, overage_data, alpha_data, fit_past, dataset_id, path):
    """ Train and evaluate the ETS model and saves model, hyperparameters and profit
    
    Parameters
    ----------
    y_train : training feature data
    y_val : validation feature data
    y_test : testing feature data
    underage_data : underage costs
    overage_data : overage costs
    alpha_data : substitution rates
    fit_past : number of past periods to fit
    dataset_id : dataset identifier
    path : path to save the model
    """
    # Initialize for measurement of memory usage and elapsed time
    monitor = MemoryMonitor(interval=1)  # monitor every second
    monitor.start()
    start = datetime.datetime.now()

    # ETS Forecasting:
    load_cost_structure(alpha_input=alpha_data, underage_input=underage_data, overage_input=overage_data) # Initialize the cost structure
    # Search the best ETS model and forecast the test set
    results_dct = ets_forecast(y_train=y_train, y_val=y_val, y_test_length=y_test.shape[0], fit_past=fit_past)
    # Evaluate the results of the ETS model
    profit_single_ets, profit_multi_ets = ets_evaluate(y_test=y_test, results_dct=results_dct)

    # Measure memory usage and elapsed time
    monitor.stop() # stop the average memory monitor
    avg_memory = monitor.average_memory_usage() # read the average memory monitor
    peak_memory = monitor.peak_memory_usage() # read the peak memory monitor
    end = datetime.datetime.now() # stop the time monitor
    elapsed = (end-start).total_seconds()
    n_products = y_test.shape[1]
    elapsed_single = elapsed/n_products
    avg_memory_single = avg_memory/n_products
    peak_memory_single = peak_memory/n_products

    # Save the model, hyperparameters, profit, time and memory usage
    save_model(model=results_dct, hyperparameter=None, profit=profit_single_ets, elapsed=elapsed_single, peak_memory=peak_memory_single, avg_memory=avg_memory_single, dataset_id=dataset_id, path=path, name='ETS_sinlge')
    save_model(model=results_dct, hyperparameter=None, profit=profit_multi_ets, elapsed=elapsed, peak_memory=peak_memory, avg_memory=avg_memory, dataset_id=dataset_id, path=path, name='ETS_multi')


def save_model(model, hyperparameter, profit, elapsed, peak_memory, avg_memory, dataset_id, path, name):
    """ Save the model, hyperparameters and profits in a pickle file

    Parameters
    ----------
    model : model to save
    hyperparameter : hyperparameters of the model
    profit_1 : profit 
    profit_2 : profit 
    dataset_id : dataset identifier
    path : path to save the model
    name : name of the model
    """
    # Ensure the path exists
    if not os.path.exists(path):
        os.makedirs(path)

    path = path + "/"

    # Create dictionaries to save the model and hyperparameters
    data_model = {
        'model': model
    }
    data_meta = {
        'hyperparameter': hyperparameter,
        'profit': profit,
        'elapsed_time': elapsed,
        'peak_memory': peak_memory,
        'avg_memory': avg_memory
    }
    # Create the file names and paths
    file_name_meta = dataset_id +"_"+ name + '_meta.pkl'
    path_name_meta = str(path) + file_name_meta
    file_name_model = dataset_id +"_"+ name + '_model.pkl'
    path_name_model = str(path) + file_name_model
    # Pickle the dictionaries into files
    with open(path_name_meta, 'wb') as f:
        pickle.dump(data_meta, f)
    with open(path_name_model, 'wb') as f:
        pickle.dump(data_model, f)

############################################################## Data Generator ############################################################

class HDF5DataGenerator(Sequence):
    def __init__(self, file_path, X_dataset_name, y_dataset_name, batch_size=32, shuffle=True):
        self.file_path = file_path
        self.X_dataset_name = X_dataset_name
        self.y_dataset_name = y_dataset_name
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        with h5py.File(self.file_path, 'r') as f:
            self.data_len = len(f[self.X_dataset_name])
        self.indexes = np.arange(self.data_len)
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(self.data_len / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__data_generation(batch_indexes)
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_indexes):
        with h5py.File(self.file_path, 'r') as f:
            X = f[self.X_dataset_name][batch_indexes]
            y = f[self.y_dataset_name][batch_indexes]
        return X, y


############################################################## Memory Monitoring ############################################################

class MemoryMonitor:
    """ Class to monitor the memory usage of the current process on a separate thread """
    def __init__(self, interval=1):
        """ Initialize the memory monitor """
        self.interval = interval # interval of measuring in seconds
        self.memory_usages = []
        self.running = False
        self.thread = None
        self.process = psutil.Process()

    def _monitor(self):
        """ Monitor the memory usage of the current process in the given interval"""
        while self.running:
            mem_info = self.process.memory_info()
            self.memory_usages.append(mem_info.rss)
            time.sleep(self.interval)

    def start(self):
        """ Start the memory monitoring thread """
        self.running = True
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()

    def stop(self):
        """ Stop the memory monitoring thread """
        self.running = False
        if self.thread:
            self.thread.join()

    def average_memory_usage(self):
        """ Calculate the average memory usage in megabytes over the runing time"""
        if not self.memory_usages:
            return 0
        average_usage_bytes = sum(self.memory_usages) / len(self.memory_usages)
        return average_usage_bytes / (1024 * 1024)  # Convert to megabytes
    
    def peak_memory_usage(self):
        """ Calculate the peak memory usage in megabytes over the runing time"""
        if not self.memory_usages:
            return 0
        peak_usage_bytes = max(self.memory_usages)
        return peak_usage_bytes / (1024 * 1024)  # Convert to megabytes