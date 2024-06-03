# Data manipulation libraries
import numpy as np
import pandas as pd

# Scikit-learn libraries for data preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Scikit-learn libraries for model selection and evaluation
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, train_test_split

# Scikit-learn library for pipeline creation
from sklearn.pipeline import Pipeline

# TensorFlow and Keras libraries for model creation and training
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.layers import Input

# Scikit-learn wrapper for Keras
from scikeras.wrappers import KerasRegressor

# pulp for mathematical optimization
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpBinary, LpStatus, PULP_CBC_CMD

from typing import Tuple
import xgboost as xgb
import optuna
from optuna_integration.keras import KerasPruningCallback


import gurobipy as gp
from gurobipy import GRB

from scipy.stats import norm

import os
from keras.utils import get_custom_objects
import datetime
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
import itertools

alpha = []
underage = []
overage = []

######################## Environment Setup Functions #####################################################################    

def create_environment():
    """ Create the environment for the newsvendor problem"""

    # Set the environment variables for Gurobi
    os.environ['GRB_LICENSE_FILE'] = '/pfs/data5/home/ma/ma_ma/ma_elanza/test_dir/gurobi.lic'
    #os.environ['TF_ENABLE_ONEDNN_OPTS']=0


######################## Data Handling Functions ############################################################

def load_data(path, multi=False):
    """ Load  data for the newsvendor problem from specified location 

    Parameters
    ---------
    path : str
    multi : bool        - if True, all products are considered, if False, only product 1 is considered
    test_size : float   - proportion of the dataset to include in the test split

    Returns
    ---------
    raw_data : np.array
    """ 
    # Load Data
    raw_data = pd.read_csv(path)    

    # Select only one product if multi == False
    if multi == False:
        # Select only columns with product_1_demand or not demand (features)
        selected_columns = raw_data.columns[raw_data.columns.str.contains('product_1_demand') | ~raw_data.columns.str.contains('demand')]
        raw_data = raw_data[selected_columns]
    return raw_data

def preprocess_data(raw_data):
    """ Preprocess the data for the newsvendor problem
    
    Parameters
    ---------
    raw_data : pd.dataframe

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


######################### Newsvendor related Functions ########################################################

def nvps_profit(demand, q, alpha, u, o):
    """ Profit function of the newsvendor under substitution
    
    Parameters
    ---------
    demand : np.array
        actual demand, shape (T, N_PRODUCTS)
    q : np.array
        predicted orders, shape (T, N_PRODUCTS)
    alpha: np.array
        substitution rates, shape (N_PRODUCTS, N_PRODUCTS)

    Returns
    ---------
    profits: np.array
        Profits by period, shape (T,1)
    """
    if demand.shape[1] == 1:
        q = np.maximum(0., q)

        if len(q.shape) == 1:
            q = q.reshape(-1, 1)

        profits = np.sum(np.minimum(q,demand)*u - np.maximum(demand-q, 0.)*(u)-np.maximum(q-demand, 0.)*(o))
    else:
        q = np.maximum(0., q) # make sure orders are non-negative
        demand_s = demand + np.matmul(np.maximum(demand-q, 0.), alpha) # demand including substitutions
        profits = np.matmul(q, u) - np.matmul(np.maximum(q-demand_s, 0.), (u+o)) # period-wise profit (T x 1)
    return profits

def solve_MILP(d, alpha, u, o, n_prods, n_threads=1):
    """ helper function that solves the mixed-integer linear program (MILP) of the multi-product newsvendor problem under substitution (cf. slides)
    
    Parameters
    -----------
    d : np.array
        Demand samples of shape (n, N_PRODUCTS), where n denotes the number of samples
    alpha : np.array
        Substitution rates, shape (N_PRODUCTS, N_PRODUCTS)
    u : np.array
        underage costs, shape (1, N_PRODUCTS)
    o : np.array
        overage costs, shape (1, N_PRODUCTS)
    n_prods : int
        number of products
    n_threads : int
        number of threads

    Returns
    ----------
    orders: np.array
        Optimal orders, of shape (1, N_PRODUCTS)
    model.status : int
        Gurobi status code (see https://www.gurobi.com/documentation/current/refman/optimization_status_codes.html)
    """
    u = u.T
    o = o.T

    hist = d.shape[0] # number of demand samples 

    # compute upper bounds M
    d_min = np.min(d, axis=0)
    d_max = d + np.matmul(d, alpha)
    M = np.array(np.max(d_max, axis=0)[0]-d_min)

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
    orders = np.array([[q[p].x for p in range(n_prods)]])
    return orders, model.status

def solve_MILP_1(d, alpha, u, o, n_threads=1):

    """ helper function that solves the mixed-integer linear program (MILP) of the multi-product newsvendor problem under substitution (cf. slides)
    
    Parameters
    -----------
    d : np.array
        Demand samples of shape (T, N_PRODUCTS), where T denotes the number of samples
    alpha : np.array
        Substitution rates, shape (N_PRODUCTS, N_PRODUCTS)
    u : np.array
        underage costs, shape (1, N_PRODUCTS)
    o : np.array
        overage costs, shape (1, N_PRODUCTS)

    n_threads : int
        number of threads

    Returns
    ----------
    orders: np.array
        Optimal orders, of shape (1, N_PRODUCTS)
    model.status : int
        Gurobi status code (see https://www.gurobi.com/documentation/current/refman/optimization_status_codes.html)
    """
    u = u.T
    o = o.T

    n_prods = d.size # number of products
    hist = 1 #d.shape[0] # number of demand samples 

    # compute upper bounds M
    d_min = np.min(d, axis=0)
    d_max = d + np.matmul(d, alpha)
    M = np.array(np.max(d_max, axis=0)[0]-d_min)

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
    orders = np.array([[q[p].X for p in range(n_prods)]])
    return orders, model.status

def solve_complex_newsvendor_seperate(y_train, y_train_pred, y_test_pred, u, o, alpha, scenario_size = 10, n_threads=40):
    """Solve the complex newsvendor problem in a parametric and a non-parametric way.

    Parameters
    ----------
    y_train : np.array
        Demand data for training, shape (T, N_PRODUCTS)
    y_train_pred : np.array
        Demand predictions for training, shape (T, N_PRODUCTS)
    y_test_pred : np.array
        Demand predictions for testing, shape (T, N_PRODUCTS)
    u : np.array
        Underage costs, shape (1, N_PRODUCTS)
    o : np.array
        Overage costs, shape (1, N_PRODUCTS)
    alpha : np.array
        Substitution rates, shape (N_PRODUCTS, N_PRODUCTS)
    scenario_size : int
        Number of scenarios to sample for the approaches
    n_threads : int
        Number of threads for the optimization

    Returns
    ----------
    final_order_quantities_parametric : np.array
        Final order quantities for each week in data_test, parametric approach
    final_order_quantities_non_parametric : np.array
        Final order quantities for each week in data_test, non-parametric approach
    """
    # Initialize an empty list to store the final order quantities
    final_order_quantities_parametric = []
    final_order_quantities_non_parametric = []

    forecast_error = y_train - y_train_pred     #  (T,n)
    forecast_error_std = forecast_error.std(axis=0) # (n,)

    # Loop over each week in data_test
    for row in y_test_pred: # row (n,)

        # Initialize arrays of shape (scenario_size, n)
        demand_scenarios_parametric = np.zeros((scenario_size, len(row)))
        demand_scenarios_non_parametric = np.zeros((scenario_size, len(row)))

        # Fill the arrays
        for i in range(scenario_size):
            demand_scenarios_parametric[i] = row + np.random.normal(loc=0, scale=forecast_error_std)
            random_row = forecast_error[np.random.randint(forecast_error.shape[0])]
            demand_scenarios_non_parametric[i] = row + random_row

        # Initialize a list to store the solutions for each scenario
        number_products= len(row)
        saa_solutions_p = []
        saa_solutions_np = []
        
        # For each demand scenario, solve the newsvendor problem
        for demand in demand_scenarios_parametric:
            # Calculate the solution for this scenario
            demand = demand.reshape(1,-1)
            solution, status = solve_MILP(d=demand, alpha=alpha, u=u, o=o, n_prods=number_products, n_threads=n_threads)
            # Store the solution for this scenario
            saa_solutions_p.append(solution)

        for demand in demand_scenarios_non_parametric:
            # Calculate the solution for this scenario
            demand = demand.reshape(1,-1)
            solution, status = solve_MILP(d=demand, alpha=alpha, u=u, o=o, n_prods=number_products, n_threads=n_threads)
            # Store the solution for this scenario
            saa_solutions_np.append(solution)
        
        # Average the solutions to get the final allocation
        final_allocation_p = np.mean(saa_solutions_p, axis=0)
        final_allocation_np = np.mean(saa_solutions_np, axis=0)

        # Store the final order quantities
        final_order_quantities_parametric.append(final_allocation_p)
        final_order_quantities_non_parametric.append(final_allocation_np)

    return final_order_quantities_parametric, final_order_quantities_non_parametric

def solve_basic_newsvendor_seperate(y_train, y_train_pred, y_test_pred, u, o, scenario_size = 10, n_threads=40):
    """Solve the basic newsvendor problem in a parametric and a non-parametric way.

    Parameters
    ----------
    y_train : np.array
        Demand data for training, shape (T, N_PRODUCTS)
    y_train_pred : np.array
        Demand predictions for training, shape (T, N_PRODUCTS)
    y_test_pred : np.array
        Demand predictions for testing, shape (T, N_PRODUCTS)
    u : np.array
        Underage costs, shape (1, N_PRODUCTS)
    o : np.array
        Overage costs, shape (1, N_PRODUCTS)
    scenario_size : int
        Number of scenarios to sample for the approaches
    n_threads : int
        Number of threads for the optimization

    Returns
    ----------
    final_order_quantities_parametric : np.array
        Final order quantities for each week in data_test, parametric approach
    final_order_quantities_non_parametric : np.array
        Final order quantities for each week in data_test, non-parametric approach
    """

    critical_ratio = u / (u + o)
    
     # Initialize an empty list to store the final order quantities
    final_order_quantities_parametric = []
    final_order_quantities_non_parametric = []
    y_train = y_train.reshape(-1,1) 
    y_train_pred = y_train_pred.reshape(-1,1)
    forecast_error = y_train - y_train_pred     #  (T,n)
    forecast_error_std = forecast_error.std(axis=0) # (n,)
  
    
    # Flatten the forecast_error
    forecast_error_flattened = np.ravel(forecast_error)

    # Loop over each week in data_test
    for i in range(len(y_test_pred)):

        # Create Demand Scenarios for this week
        demand_scenarios_parametric = y_test_pred[i] + np.random.normal(loc=0, scale=forecast_error_std, size=scenario_size)
        
        demand_scenarios_non_parametric = y_test_pred[i] + np.random.choice(forecast_error_flattened, size=scenario_size) 

        # Initialize a list to store the solutions for each scenario
        saa_solutions_p = []
        saa_solutions_np = []

        # For each demand scenario, solve the newsvendor problem
        for demand in demand_scenarios_parametric:
            # Calculate the solution for this scenario
            solution = norm.ppf(critical_ratio,loc=demand,scale=forecast_error_std)
            # Store the solution for this scenario
            saa_solutions_p.append(solution)

        for demand in demand_scenarios_non_parametric:
            # Calculate the solution for this scenario
            solution = norm.ppf(critical_ratio,loc=demand,scale=forecast_error_std)
            # Store the solution for this scenario
            saa_solutions_np.append(solution)

        # Average the solutions to get the final allocation
        final_allocation_p = np.mean(saa_solutions_p, axis=0)
        final_allocation_np = np.mean(saa_solutions_np, axis=0)

        # Store the final order quantities
        final_order_quantities_parametric.append(final_allocation_p)
        final_order_quantities_non_parametric.append(final_allocation_np)

    return final_order_quantities_parametric, final_order_quantities_non_parametric

############################### ETS Functions ###########################################################################

def ets_forecast( y_train, y_val, y_test_length, verbose=0, fit_past = 12*7):
    """Forecast the demand using the ETS model

    Parameters
    ----------
    y_train : np.array
        Demand data for training, shape (T, N_PRODUCTS)
    y_val : np.array
        Demand data for validation, shape (T, N_PRODUCTS)
    y_test_length : int
        Number of samples in the test set
    verbose : int
        Verbosity level
    fit_past : int
        Number of samples in the demand distribution estimate

    Returns
    ----------
    results_dct : dict
        Dictionary containing the results for each product
    elapsed : float
        Elapsed time
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

    
    best_config, best_rmse, best_mape, best_message = None, np.inf, np.inf, None
    best_rmse = np.inf
    results_dct = {}
    
    print('Starting model selection.\n')
    elapsed = 0 # elapsed time in seconds

    # loop over products
    for i in range(N_PRODUCTS):
        start_i = datetime.datetime.now() # computation start for product i
        
        if verbose>0:
            print('Starting product {}/{}.'.format(i+1, N_PRODUCTS)) # console update
            
        results_dct[i] = {} # initialize result dict for product i
        target = np.append(y_train[:,i], y_val[:, i]) # train and val targets for i
        
        # loop through configurations
        for n_config, config in enumerate(model_configs, 1):
            if verbose>0:
                print('Starting config {}/{}.'.format(n_config, len(model_configs))) # console update
            preds = np.array([]) # variable in which validation set predictions are stored
            try:
                # loop through validation timesteps, fit and predict on-day-ahead
                for t in range(0, N_VAL, timesteps):
                    model = ETSModel(target[N_TRAIN+t-fit_past:N_TRAIN+t], *config)
                    model = model.fit()
                    preds = np.append(preds, np.maximum(0, model.forecast(timesteps)))
                    if verbose==2:
                        print(t, end=',') # console update
                if verbose==2:
                    print('\n')
                # after evaluation is completed, compute RMSE and MAPE on validation set
                target_val = target[N_TRAIN:N_TRAIN+N_VAL]
                print(y_val.shape, target_val.shape, preds.shape)
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
        
        results_dct[i] = (best_config, best_rmse, best_mape, best_test_pred, best_message) # save results for product "i" and configuration "config"
        elapsed_i = (datetime.datetime.now()-start_i).total_seconds() # record time
        elapsed += elapsed_i
        if verbose>0:
            print('Elapsed for product {}/{}: {}.'.format(i+1, N_PRODUCTS, elapsed_i)) # console update
            print('\n\n')
    print('Total elapsed: {}.'.format(elapsed))

    return results_dct, elapsed

def ets_evaluate(y_test, results_dct, underage, overage, alpha, verbose=0):
    """Evaluate the ETS model

    Parameters
    ----------
    y_test : np.array
        Demand data for testing, shape (T, N_PRODUCTS)
    results_dct : dict
        Dictionary containing the results for each product
    underage : np.array
        Underage costs, shape (1, N_PRODUCTS)
    overage : np.array
        Overage costs, shape (1, N_PRODUCTS)
    alpha : np.array
        Substitution rates, shape (N_PRODUCTS, N_PRODUCTS)
    verbose : int
        Verbosity level

    Returns
    ----------
    profit_ets_single : float
        Profit for the single product case
    profit_ets_multi : float
        Profit for the multi-product case
    """
    N_PRODUCTS = y_test.shape[1] # number of products
    N_TEST = y_test.shape[0] # number of test samples

    test_pred = np.zeros((N_TEST, N_PRODUCTS))  # Assuming N_TEST is the number of test samples
    
    for i in range(N_PRODUCTS):
        test_pred[:, i] = results_dct[i][3]

    y_test_single = y_test[:,0].reshape(-1, 1)
    test_pred_single = test_pred[:,0].reshape(-1, 1)

    underage_single = underage[0,0]
    overage_single = overage[0,0]

    profit_ets_single = np.mean(nvps_profit(y_test_single, test_pred_single, None, underage_single, overage_single))
    profit_ets_multi = np.mean(nvps_profit(y_test, test_pred, alpha, underage, overage))

    return  profit_ets_single, profit_ets_multi
    
    

    

######################### Neural Network Functions ######################################################################

def make_nvps_loss(alpha, underage, overage):
    """ Create a custom loss function for the newsvendor problem under substitution
    
    Parameters
    ---------
    alpha : np.array
        substitution matrix
    underage : np.array
        underage costs
    overage : np.array
        overage costs

    Returns
    ---------
    nvps_loss : function
        Custom loss function
    """
    # transofrm the alpha, u, o to tensors
    underage = tf.convert_to_tensor(underage, dtype=tf.float32) #underage costs
    overage = tf.convert_to_tensor(overage, dtype=tf.float32) #overage costs
    alpha = tf.convert_to_tensor(alpha, dtype=tf.float32) #substitution matrix

    # define the loss function
    @tf.autograph.experimental.do_not_convert
    def nvps_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        q = tf.maximum(y_pred, 0.)

        # Calculate the demand increase for each product due to substitutions from other products
        demand_increase = tf.matmul( tf.maximum(0.0, y_true - y_pred),alpha)
        # Adjusted demand is the original demand plus the increase due to substitutions
        adjusted_demand = y_true + demand_increase
        # Calculate the profits
        profits = tf.matmul(q,underage) - tf.matmul(tf.maximum(0.0,q - adjusted_demand), underage+overage)

        return -tf.math.reduce_mean(profits)
        
    return nvps_loss

def make_nvp_loss(underage, overage):
    """ Create a custom loss function for the newsvendor problem without substitution
    
    Parameters
    ---------
    underage : np.array
        underage costs
    overage : np.array
        overage costs

    Returns
    ---------
    nvp_loss : function
        Custom loss function
    """
    q = underage / (underage + overage)
    @tf.autograph.experimental.do_not_convert
    def nvp_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        error = y_true - y_pred
        return tf.keras.backend.mean(tf.maximum(q*error, (q-1)*error), axis=-1)
    return nvp_loss

def create_NN_model(n_hidden, n_neurons, activation, input_shape, learning_rate, custom_loss, output_shape, seed=42): 
    """ Build a neural network model with the specified architecture and hyperparameters     """
    model = Sequential()
    model.add(Input(shape=(input_shape,)))
    for layer in range(n_hidden):
        model.add(Dense(n_neurons, activation=activation))
    model.add(Dense(output_shape))
    # set seed for reproducability
    tf.random.set_seed(seed)
    np.random.seed(seed)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss=custom_loss, optimizer=optimizer, metrics=None)
    return model

def create_NN_multi(n_hidden, n_neurons, activation, input_shape, learning_rate, output_shape, seed=42):
    global alpha, underage, overage
    loss = make_nvps_loss(alpha=alpha, underage=underage, overage=overage)
    return create_NN_model(n_hidden, n_neurons, activation, input_shape, learning_rate, loss, output_shape, seed)

def create_NN_single(n_hidden, n_neurons, activation, input_shape, learning_rate, output_shape, seed=42):
    global alpha, underage, overage
    loss = make_nvp_loss(underage=underage, overage=overage)
    return create_NN_model(n_hidden, n_neurons, activation, input_shape, learning_rate, loss, output_shape, seed)

def create_NN_basic(n_hidden, n_neurons, activation, input_shape, learning_rate, output_shape, seed=42):
    loss = tf.keras.losses.MeanSquaredError()
    get_custom_objects().update({'custom_loss': loss})
    return create_NN_model(n_hidden, n_neurons, activation, input_shape, learning_rate, loss, output_shape, seed)

def tune_NN_model_optuna(X_train, y_train, X_val, y_val, alpha_input, underage_input, overage_input, patience=10, multi=True, integrated=True, verbose=0, seed=42, threads=40, trials=100):
    """ Tune a neural network model on the given training data with early stopping using Optuna.
    
    Parameters
    --------------
    X_train : np.array
        training feature data (samples, features)
    y_train : np.array
        training targets (samples, N_PRODUCTS)
    X_val : np.array
        validation feature data (samples, features)
    y_val : np.array
        validation targets (samples, N_PRODUCTS)
    alpha_input : np.array
        Substitution rates, shape (N_PRODUCTS, N_PRODUCTS)
    underage_input : np.array
        underage costs, shape (1, N_PRODUCTS)
    overage_input : np.array
        overage costs, shape (1, N_PRODUCTS)    
    patience : int
        number of epochs without improvement before stopping
    multi : bool
        if True, all products are considered, if False, only product 1 is considered
    integrated : bool
        if True, the optimization is integrated, if False, the optimization is separate
    verbose : int
        keras' verbose parameter for silent / verbose model training
    seed : int
        random seed (affects mainly model initialization, set for reproducable results)

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
    if alpha_input is not None:
        alpha = alpha_input
    underage = underage_input
    overage = overage_input

    output_shape = y_train.shape[1] #N_PRODUCTS
    input_shape = X_train.shape[1] #N_FEATURES
    
    def objective(trial):
        # define the hyperparameters space
        n_hidden = trial.suggest_int('n_hidden', 0, 10)
        n_neurons = trial.suggest_int('n_neurons', 1, 30)
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        epochs = trial.suggest_int('epochs', 10, 50)
        activation = trial.suggest_categorical('activation', ['relu', 'sigmoid', 'tanh'])

        # create a neural network model with basic hyperparameters
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience)

        
        # construct loss function based on the number of products
        if not integrated:
            model_ANN = KerasRegressor(model=create_NN_basic, n_hidden=n_hidden, n_neurons=n_neurons, activation=activation,
                                    input_shape=input_shape, learning_rate=learning_rate, output_shape=output_shape, 
                                    seed=seed, verbose=verbose, callbacks=[early_stopping])
        elif not multi and integrated:
            model_ANN = KerasRegressor(model=create_NN_single, n_hidden=n_hidden, n_neurons=n_neurons, activation=activation,
                                input_shape=input_shape, learning_rate=learning_rate, output_shape=output_shape, 
                                seed=seed, verbose=verbose, callbacks=[early_stopping])
        elif multi and integrated: 
            model_ANN = KerasRegressor(model=create_NN_multi, n_hidden=n_hidden, n_neurons=n_neurons, activation=activation,
                                input_shape=input_shape, learning_rate=learning_rate, output_shape=output_shape, 
                                seed=seed, verbose=verbose, callbacks=[early_stopping])
        else:
            raise ValueError('Invalid Configuration')
        

        #pruning_callback = KerasPruningCallback(trial, 'val_loss')
        model_ANN.fit(X_train, y_train, validation_data=(X_val, y_val),  epochs=epochs, batch_size=batch_size, verbose=verbose) #, callbacks=[pruning_callback]

        # make predictions on validation set and compute profits
        q_val = model_ANN.predict(X_val)

        # If integrated, we can use the profit function, 
        #       otherwise we use the negative absolute error (otherwise we would "cheat")
        if integrated:
            result = np.mean(nvps_profit(y_val, q_val, alpha, underage, overage))
        else:
            result = -np.abs(np.mean(q_val-y_val))

        return result

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=trials, n_jobs=threads)

    # Get the best parameters and best estimator
    best_params = study.best_params
    best_estimator = None

    hyperparameter = [best_params['n_hidden'], best_params['n_neurons'],best_params['learning_rate'], 
                    best_params['epochs'], patience, best_params['batch_size'], best_params['activation']]
    
    return best_estimator, hyperparameter, study.best_value

def train_NN_model(hp, X_train, y_train, X_val, y_val, alpha, underage, overage, multi=True, integrated=True, verbose=0, seed=42):
    """ Train a network on the given training data with early stopping.
    
    Parameters
    --------------
    hp : list or tupl
        hyperparameters in the following order: hidden_nodes, lr, max_epochs, patience, batch_size
    X_train : np.array
        training feature data
    y_train : np.array
        training targets
    X_train : np.array
        validation feature data
    y_train : np.array
        validation targets
    alpha : np.array
        Substitution rates, shape (N_PRODUCTS, N_PRODUCTS)
    u : np.array
        underage costs, shape (1, N_PRODUCTS)
    o : np.array
        overage costs, shape (1, N_PRODUCTS)
    verbose : int
        keras' verbose parameter for silent / verbose model training
    seed : int
        random seed (affects mainly model initialization, set for reproducable results)

    Returns
    ----------
    model : keras model
        Final model
    """

    # construct loss function based on the number of products
    if integrated == False:
        loss = tf.keras.losses.MeanSquaredError()
    elif (multi == False) & (integrated == True):
        loss = make_nvp_loss(underage, overage)
    elif (multi == True) & (integrated == True): 
        loss = make_nvps_loss(alpha, underage, overage)

    # extract hyperparameters, build and compile MLP
    hidden_nodes, n_neurons, lr, max_epochs, patience, batch_size, activation = hp
    mlp = create_NN_model(n_hidden=hidden_nodes, n_neurons=n_neurons, activation=activation, 
                          input_shape=X_train.shape[1], learning_rate=lr, custom_loss=loss, 
                          output_shape=y_train.shape[1], seed=seed)

    # train MLP with early stopping
    callback = EarlyStopping(monitor='val_loss', patience=patience)
    mlp.fit(X_train, y_train, epochs=max_epochs, batch_size=batch_size, validation_data=(X_val, y_val),
            verbose=verbose, callbacks=[callback])
    
    return mlp

######################### LightGBM Functions ######################################################################

def gradient(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
    if isinstance(dtrain, np.ndarray):
        dtrain = xgb.DMatrix(dtrain)
    global alpha, underage, overage
    y = dtrain.get_label().reshape(predt.shape)
    d = y + np.matmul(np.maximum(0, y - predt), alpha)
    u = np.array(underage)
    o = np.array(overage)
    u = u.T
    o = o.T
    return (-(u * np.maximum(0,d-predt) - o * np.maximum(0, predt-d))).reshape(y.size)
                
def hessian(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
    return np.ones(predt.shape).reshape(predt.size)
        
def custom_loss(predt: np.ndarray, dtrain: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    grad = gradient(predt, dtrain)
    hess = hessian(predt, dtrain)

    return grad, hess

def tune_XGB_model(X_train, y_train, X_val, y_val, alpha_input, underage_input, overage_input, patience=10, multi=True, integrated=True, verbose=0, seed=42, threads=40, trials=100):

    global alpha, underage, overage
    if alpha_input is not None:
        alpha = alpha_input
    underage = underage_input
    overage = overage_input

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


    X, y = X_train, y_train  
    Xy = xgb.DMatrix(X, label=y)
    dval = xgb.DMatrix(X_val, label=y_val)
    results = {}
    def objective(trial):
        if custom_objective != custom_loss:
            params = {
                "tree_method": "hist",
                "num_target": y.shape[1],
                "multi_strategy": multi_strategy,
                "learning_rate": trial.suggest_float("learning_rate", 0.1, 0.5),
                "max_depth": trial.suggest_int("max_depth", 2, 6),
                "n_estimators": trial.suggest_int("n_estimators", 100, 140),    #????????
                "subsample": trial.suggest_float("subsample", 0.3, 0.9),
                "quantile_alpha": quantile,
                "objective": custom_objective,
            }
            booster = xgb.train(
                params,
                dtrain=Xy,
                num_boost_round=128,
                #obj=custom_objective,
                evals=[(dval, "val")],
                evals_result=results,
                early_stopping_rounds=patience,
            )
        else:
            params = {
                "tree_method": "hist",
                "num_target": y.shape[1],
                "multi_strategy": multi_strategy,
                "learning_rate": trial.suggest_float("learning_rate", 0.1, 0.5),
                "max_depth": trial.suggest_int("max_depth", 2, 6),
                "n_estimators": trial.suggest_int("n_estimators", 100, 140),    #????????
                "subsample": trial.suggest_float("subsample", 0.3, 0.9),
                "quantile_alpha": quantile,
                #"objective": custom_objective,
            }
            booster = xgb.train(
            params,
                dtrain=Xy,
                num_boost_round=128,
                obj=custom_objective,
                evals=[(dval, "val")],
                evals_result=results,
                early_stopping_rounds=patience,
            )
       
         # make predictions on validation set and compute profits
        val_set = xgb.DMatrix(X_val)
        q_val = booster.predict(val_set)

        # If integrated, we can use the profit function, 
        #       otherwise we use the negative absolute error (otherwise we would "cheat")
        if integrated:
            result = np.mean(nvps_profit(y_val, q_val, alpha, underage, overage))
        else:
            result = -np.abs(np.mean(q_val-y_val))

        return result

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trials, n_jobs=threads)  

    trial = study.best_trial
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Get the best parameters
    best_params = study.best_trial.params

    # Add the fixed parameters
    if custom_objective != custom_loss:
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
            #obj=custom_objective,
            evals=[(dval, "val")],
        )
    else:
        final_booster = xgb.train(
            best_params,
            dtrain=Xy,
            num_boost_round=128,
            obj=custom_objective,
            evals=[(dval, "val")],
        )


    return final_booster, best_params, results

def train_XGB_model(hyperparameter, X_train, y_train, X_val, y_val, alpha_data, underage_data, overage_data):

    global alpha, underage, overage
    alpha = alpha_data
    underage = underage_data.flatten()
    overage = overage_data.flatten()

    # Train the final model
    Xy = xgb.DMatrix(X_train, label=y_train)
    results = {}
    
    final_booster = xgb.train(
        hyperparameter,
        dtrain=Xy,
        num_boost_round=128,
        obj=custom_loss,
        evals=[(Xy, "Train")],
        custom_metric=newsvendorRMSE
    )
    return final_booster, results




