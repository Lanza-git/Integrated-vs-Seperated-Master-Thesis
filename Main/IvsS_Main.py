# General imports
import subprocess
import sys

from IvsS_Utils import load_packages

load_packages()

import numpy as np

# custom functions and constants
from IvsS_Utils import load_data, preprocess_data, split_data, nvps_profit, solve_MILP, create_environment
from IvsS_Utils import train_NN_model, tune_NN_model_optuna, solve_basic_newsvendor_seperate, solve_complex_newsvendor_seperate


####################################### Constants ##############################################################################

# Parameters for multi-item newsvendor problem
prices = np.array([0.3, 0.5, 0.6, 0.5, 0.5, 0.5]) #price data
prices = prices.reshape(6,1)
costs = np.array([0.06, 0.06, 0.06, 0.06, 0.06, 0.06]) #cost data
costs = costs.reshape(6,1)
salvages = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01]) #salvage data
salvages = salvages.reshape(6,1)
underage_data = prices - costs 
overage_data = costs - salvages 
underage_data_single = underage_data[0,0]
overage_data_single = overage_data[0,0]



alpha_data = np.array([             #alpha data
    [0.0, 0.1, 0.05, 0.1, 0.05, 0.1],
    [0.15, 0.0, 0.1, 0.05, 0.05, 0.05],
    [0.1, 0.2, 0.0, 0.05, 0.1, 0.05],
    [0.05, 0.05, 0.05, 0.0, 0.15, 0.2],
    [0.1, 0.05, 0.15, 0.2, 0.0, 0.05],
    [0.05, 0.1, 0.05, 0.15, 0.1, 0.0]
])

####################################### Functions ##############################################################################

if __name__ == "__main__":
    
    create_environment()

    """
    TODOS
    - IOA complex not working - resolved
    - SOA is cheating because of optuna - Resolved: check with Xander
    - XGBoost - resolved
    - Baseline - ARMA and ETS
    
    
    
    """
    
    # Neural network - Complex
    path = "data.csv"
    
    multi_data = load_data(path, True)
    multi_feature_data, multi_target_data = preprocess_data(multi_data)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(multi_feature_data, multi_target_data)

    X_train = X_train[:10]
    y_train = y_train[:10]
    X_val = X_val[:2]
    y_val = y_val[:2]

    X_test = X_test[:2]
    y_test = y_test[:2]
    
    
    # Integrated Optimization Approach:
    best_estimator, hyperparameter, val_profit = tune_NN_model_optuna(X_train, y_train, X_val, y_val, alpha_data, underage_data, overage_data)
    model_ANN_complex = train_NN_model(hyperparameter, X_train, y_train, X_val, y_val, alpha_data, underage_data, overage_data)
    target_prediction_ANN = model_ANN_complex.predict(X_test)
    profit_complex_ANN_IOA = np.mean(nvps_profit(y_test, target_prediction_ANN, alpha_data, underage_data, overage_data))
    

    print("Step 1")
    
    # Seperate Optimization Approach:
    best_estimator, hyperparameter, val_profit = tune_NN_model_optuna(X_train, y_train, X_val, y_val, alpha_data, underage_data, overage_data, multi = True, integrated = False)
    model_ANN_complex = train_NN_model(hyperparameter, X_train, y_train, X_val, y_val, alpha_data, underage_data, overage_data, multi = True,  integrated = False)
    target_prediction_ANN = model_ANN_complex.predict(X_test)
    train_prediction_ANN = model_ANN_complex.predict(X_train)
    orders_scp_ann, orders_scnp_ann = solve_complex_newsvendor_seperate(y_train=y_train, y_train_pred=train_prediction_ANN, y_test_pred=target_prediction_ANN, u=underage_data, o=overage_data, alpha=alpha_data)
    profit_scp_ANN = np.mean(nvps_profit(y_test, orders_scp_ann, alpha_data, underage_data, overage_data))
    profit_scnp_ANN = np.mean(nvps_profit(y_test, orders_scnp_ann, alpha_data, underage_data, overage_data))
    
    print("Step 2")
    
    # Neural network - Simple
    single_data = load_data(path, False)
    single_feature_data, single_target_data = preprocess_data(single_data)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(single_feature_data, single_target_data)
   
    # Integrated Optimization Approach:
    best_estimator, hyperparameter, val_profit = tune_NN_model_optuna(X_train, y_train, X_val, y_val, None, underage_data_single, overage_data_single, multi = False)
    model_ANN_simple = train_NN_model(hyperparameter, X_train, y_train, X_val, y_val, None, underage_data_single, overage_data_single, multi = False)
    target_prediction_ANN = model_ANN_simple.predict(X_test)
    profit_simple_ANN_IOA = np.mean(nvps_profit(y_test, target_prediction_ANN, None, underage_data_single, overage_data_single))
    
    print("Step 3")

    # Seperate Optimization Approach:
    best_estimator, hyperparameter, val_profit = tune_NN_model_optuna(X_train, y_train, X_val, y_val, None, underage_data_single, overage_data_single, multi = False, integrated = False)
    model_ANN_simple = train_NN_model(hyperparameter, X_train, y_train, X_val, y_val, None, underage_data_single, overage_data_single, multi = False, integrated = False)
    target_prediction_ANN = model_ANN_simple.predict(X_test)
    train_prediction_ANN = model_ANN_simple.predict(X_train)
    orders_ssp_ann, orders_ssnp_ann = solve_basic_newsvendor_seperate(y_train=y_train, y_train_pred=train_prediction_ANN, y_test_pred=target_prediction_ANN, u=underage_data_single, o=overage_data_single)
    profit_ssp_ANN = np.mean(nvps_profit(y_test, orders_ssp_ann, alpha_data, underage_data_single, overage_data_single))
    profit_ssnp_ANN = np.mean(nvps_profit(y_test, orders_ssnp_ann, alpha_data, underage_data_single, overage_data_single))
    
    # Print results
    print("Profit Complex ANN IOA: ", profit_complex_ANN_IOA)
    print("Profit Complex ANN SOA - parametric: ", profit_scp_ANN)
    print("Profit Complex ANN SOA - non-parametric: ", profit_scnp_ANN)
    print("Profit Simple ANN IOA: ", profit_simple_ANN_IOA)
    print("Profit Simple ANN SOA - parametric: ", profit_ssp_ANN)
    print("Profit Simple ANN SOA - non-parametric: ", profit_ssnp_ANN)
    