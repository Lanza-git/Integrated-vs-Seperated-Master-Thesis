# General imports
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install('pandas')
install('scikit-learn')
install('tensorflow')
install('scikeras')
install('numpy')
install('pulp')
install('xgboost')
install('typing')
install('optuna')
install('optuna-integration')
install('gurobipy')


import numpy as np
import xgboost as xgb

# custom functions and constants
from IvsS_Utils import load_data, preprocess_data, split_data, nvps_profit, solve_MILP, create_environment
from IvsS_Utils import tune_XGB_model


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
    path = "data.csv"
    multi_data = load_data(path, True)
    multi_feature_data, multi_target_data = preprocess_data(multi_data)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(multi_feature_data, multi_target_data)
    print("Data loaded")	
    
    # Integrated Optimization Approach:
    xgb_model, params, results = tune_XGB_model(X_train, y_train, None, None, alpha_data, underage_data, overage_data)
    print("Model trained")
    xgb_result = xgb_model.predict(xgb.DMatrix(X_test, label=y_test))
    print("Model predicted")
    profit_complex_DT_IOA = np.mean(nvps_profit(y_test, xgb_result, alpha_data, underage_data, overage_data))
    print("Profit for complex model using XGBoost: ", profit_complex_DT_IOA)

    # Seperated Optimization Approach:
    xgb_model, hyperparameter_XGB_SOA_Complex, val_profit = tune_XGB_model(X_train, y_train, X_val, y_val, alpha_data, underage_data, overage_data, multi = True, integrated = False)
    target_prediction_XGB = xgb_model.predict(xgb.DMatrix(X_test, label=y_test))
    orders_XGB_complex, status_XGB_complex = solve_MILP(d=target_prediction_XGB, alpha=alpha_data, u= underage_data, o=overage_data, n_threads=40)
    profit_complex_XGB_SOA = np.mean(nvps_profit(y_test, orders_XGB_complex, alpha_data, underage_data, overage_data))

    print("Step 2: "+ str(profit_complex_XGB_SOA))

   # Neural network - Simple
    single_data = load_data(path, False)
    single_feature_data, single_target_data = preprocess_data(single_data)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(single_feature_data, single_target_data)

    # Integrated Optimization Approach:
    xgb_model, params, results = tune_XGB_model(X_train, y_train, None, None, alpha_data, underage_data, overage_data)
    xgb_result = xgb_model.predict(xgb.DMatrix(X_test, label=y_test))
    profit_simple_XGB_IOA = np.mean(nvps_profit(y_test, xgb_result, alpha_data, underage_data, overage_data))

    print("Profit for simple model using XGBoost: ", profit_simple_XGB_IOA)
    print("Step 3")

    # Seperated Optimization Approach:
    xgb_model, hyperparameter_XGB_SOA_Complex, val_profit = tune_XGB_model(X_train, y_train, X_val, y_val, alpha_data, underage_data, overage_data, multi = True, integrated = False)
    target_prediction_XGB = xgb_model.predict(xgb.DMatrix(X_test, label=y_test))
    #orders_XGB_simple, status_XGB_simple = solve_MILP_CBC(target_prediction_XGB, alpha_data, underage_data, overage_data, 40)
    profit_simple_XGB_SOA = 0# np.mean(nvps_profit(y_test, orders_XGB_complex, alpha_data, underage_data, overage_data))

    # Print results
    print("Profit Complex ANN IOA: ", profit_complex_DT_IOA)
    print("Profit Complex ANN SOA: ", profit_complex_XGB_SOA)
    print("Profit Simple ANN IOA: ", profit_simple_XGB_IOA)
    print("Profit Simple ANN SOA: ", profit_simple_XGB_SOA)


