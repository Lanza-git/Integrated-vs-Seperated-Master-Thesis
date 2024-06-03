# Load packages
#from IvsS_Utils import load_packages
def load_packages():
    # General imports
    import subprocess
    import sys


    def install(package):
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

    install('pandas')
    install('scikit-learn')
    install('scikeras')
    install('numpy')
    install('pulp')
    install('xgboost')
    install('typing')
    install('optuna')
    install('optuna-integration')
    install('gurobipy')
    install('statsmodels')
    install('tensorflow<2.13')
    install('mpi4py')

load_packages

import numpy as np
import xgboost as xgb

# custom functions and constants
from IvsS_Utils import load_data, preprocess_data, split_data, nvps_profit, solve_MILP, create_environment
from IvsS_Utils import tune_XGB_model, solve_complex_newsvendor_seperate, solve_basic_newsvendor_seperate


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

    # Decision Tree - Complex
    path = "Main/data.csv"
    multi_data = load_data(path, True)
    multi_feature_data, multi_target_data = preprocess_data(multi_data)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(multi_feature_data, multi_target_data)

    # Integrated Optimization Approach:
    xgb_model, params, results = tune_XGB_model(X_train, y_train, X_val, y_val, alpha_data, underage_data, overage_data)
    xgb_result = xgb_model.predict(xgb.DMatrix(X_test))
    profit_complex_XGB_IOA = np.mean(nvps_profit(y_test, xgb_result, alpha_data, underage_data, overage_data))
    
    print("profit complex", profit_complex_XGB_IOA)
    
    # Seperated Optimization Approach:
    xgb_model, hyperparameter_XGB_SOA_Complex, val_profit = tune_XGB_model(X_train, y_train, X_val, y_val, alpha_data, underage_data, overage_data, multi = True, integrated = False)
    target_prediction_XGB = xgb_model.predict(xgb.DMatrix(X_test))
    train_prediction_XGB = xgb_model.predict(xgb.DMatrix(X_train))
    orders_scp_XGB, orders_scnp_XGB = solve_complex_newsvendor_seperate(y_train=y_train, y_train_pred=train_prediction_XGB, y_test_pred=target_prediction_XGB, u=underage_data, o=overage_data, alpha=alpha_data)
    profit_scp_XGB = np.mean(nvps_profit(y_test, orders_scp_XGB, alpha_data, underage_data, overage_data))
    profit_scnp_XGB = np.mean(nvps_profit(y_test, orders_scnp_XGB, alpha_data, underage_data, overage_data))
    
    print("Step 2: "+ str(profit_scnp_XGB), profit_scp_XGB)
    
    # Neural network - Simple
    single_data = load_data(path, False)
    single_feature_data, single_target_data = preprocess_data(single_data)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(single_feature_data, single_target_data)
    
    # Integrated Optimization Approach:
    xgb_model, params, results = tune_XGB_model(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, alpha_input=None, underage_input=underage_data_single, overage_input=overage_data_single, multi=False, integrated=True)
    xgb_result = xgb_model.predict(xgb.DMatrix(X_test))
    profit_simple_XGB_IOA = np.mean(nvps_profit(y_test, xgb_result, None, underage_data_single, overage_data_single))

    print("Step 3")
    
    # Seperated Optimization Approach:
    xgb_model, hyperparameter_XGB_SOA_Complex, val_profit = tune_XGB_model(X_train, y_train, X_val, y_val, None, underage_data_single, overage_data_single, multi = False, integrated = False)
    target_prediction_XGB = xgb_model.predict(xgb.DMatrix(X_test))
    train_prediction_XGB = xgb_model.predict(xgb.DMatrix(X_train))  
    orders_ssp_XGB, orders_ssnp_XGB = solve_basic_newsvendor_seperate(y_train=y_train, y_train_pred=train_prediction_XGB, y_test_pred=target_prediction_XGB, u=underage_data_single, o=overage_data_single)
    profit_ssp_XGB = np.mean(nvps_profit(y_test, orders_ssp_XGB, alpha_data, underage_data_single, overage_data_single))
    profit_ssnp_XGB = np.mean(nvps_profit(y_test, orders_ssnp_XGB, alpha_data, underage_data_single, overage_data_single))
    
    # Print results
    print("Profit Complex ANN IOA: ", profit_complex_XGB_IOA)
    print("Profit Complex ANN SOA - parametric: ", profit_scp_XGB)
    print("Profit Complex ANN SOA - non-parametric: ", profit_scnp_XGB)
    print("Profit Simple ANN IOA: ", profit_simple_XGB_IOA)
    print("Profit Simple ANN SOA - parametric: ", profit_ssp_XGB)
    print("Profit Simple ANN SOA - non-parametric: ", profit_ssnp_XGB)
    