# General imports
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install('pandas')
install('scikit-learn')
install('tensorflow')
install('scikeras')
install('scipy')
install('numpy')
install('pulp')


from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.optimizers import Adam
from scipy.stats import reciprocal
import numpy as np
from scipy.stats import norm

# custom functions and constants
from IvsS_Utils import load_data, preprocess_data, split_data
from IvsS_Utils import nvps_profit, solve_MILP
from IvsS_Utils import tune_NN_model, train_NN_model


####################################### Constants ##############################################################################

# Parameters for multi-item newsvendor problem
prices = np.array([0.3, 0.5, 0.6, 0.5, 0.5, 0.5]) #price data
costs = np.array([0.06, 0.06, 0.06, 0.06, 0.06, 0.06]) #cost data
salvages = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01]) #salvage data
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

    # Neural network - Complex
    path = "data.csv"
    multi_data = load_data(path, True)
    multi_feature_data, multi_target_data = preprocess_data(multi_data)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(multi_feature_data, multi_target_data)

    print(y_test.shape)

    # Integrated Optimization Approach:
    best_estimator, hyperparameter, val_profit = tune_NN_model(X_train, y_train, X_val, y_val, alpha_data, underage_data, overage_data)
    model_ANN_complex = train_NN_model(hyperparameter, X_train, y_train, X_val, y_val, alpha_data, underage_data, overage_data)
    target_prediction_ANN = model_ANN_complex.predict(X_test)
    profit_complex_ANN_IOA = np.mean(nvps_profit(y_test, target_prediction_ANN, alpha_data, underage_data, overage_data))

    print("Step 1")

    # Seperate Optimization Approach:
    best_estimator, hyperparameter, val_profit = tune_NN_model(X_train, y_train, X_val, y_val, alpha_data, underage_data, overage_data, integrated = False)
    model_ANN_complex = train_NN_model(hyperparameter, X_train, y_train, X_val, y_val, alpha_data, underage_data, overage_data, integrated = False)
    target_prediction_ANN = model_ANN_complex.predict(X_test)
    orders_ANN_complex, status_ANN_complex = solve_MILP(target_prediction_ANN, alpha_data, underage_data, overage_data)
    profit_complex_ANN_SOA = np.mean(nvps_profit(y_test, orders_ANN_complex, alpha_data, underage_data, overage_data))

    print("Step 2")

    # Neural network - Simple
    multi_data = load_data(path, False)
    multi_feature_data, multi_target_data = preprocess_data(multi_data)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(multi_feature_data, multi_target_data)

    # Integrated Optimization Approach:
    best_estimator, hyperparameter, val_profit = tune_NN_model(X_train, y_train, X_val, y_val, alpha_data, underage_data, overage_data)
    model_ANN_simple = train_NN_model(hyperparameter, X_train, y_train, X_val, y_val, alpha_data, underage_data, overage_data)
    target_prediction_ANN = model_ANN_simple.predict(X_test)
    profit_simple_ANN_IOA = np.mean(nvps_profit(y_test, target_prediction_ANN, alpha_data, underage_data, overage_data))

    print("Step 3")

    # Seperate Optimization Approach:
    best_estimator, hyperparameter, val_profit = tune_NN_model(X_train, y_train, X_val, y_val, alpha_data, underage_data, overage_data, integrated = False)
    model_ANN_simple = train_NN_model(hyperparameter, X_train, y_train, X_val, y_val, alpha_data, underage_data, overage_data, integrated = False)
    target_prediction_ANN = model_ANN_simple.predict(X_test)
    orders_ANN_simple, status_ANN_simple = solve_MILP(target_prediction_ANN, alpha_data, underage_data, overage_data)
    profit_simple_ANN_SOA = np.mean(nvps_profit(y_test, orders_ANN_simple, alpha_data, underage_data, overage_data))

    # Print results
    print("Profit Complex ANN IOA: ", profit_complex_ANN_IOA)
    print("Profit Complex ANN SOA: ", profit_complex_ANN_SOA)
    print("Profit Simple ANN IOA: ", profit_simple_ANN_IOA)
    print("Profit Simple ANN SOA: ", profit_simple_ANN_SOA)

    print("Hyperparameter Complex ANN IOA: ", hyperparameter)
    print("Hyperparameter Complex ANN SOA: ", hyperparameter)
    print("Hyperparameter Simple ANN IOA: ", hyperparameter)
    print("Hyperparameter Simple ANN SOA: ", hyperparameter)