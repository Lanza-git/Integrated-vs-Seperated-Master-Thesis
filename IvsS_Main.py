# General imports
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install('pandas')
install('scikit-learn')
install('tensorflow')
install('scikeras')
install('lightgbm')
install('scipy')
install('numpy')
install('gurobipy')


import pandas as pd
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.optimizers import Adam
from scipy.stats import reciprocal
import numpy as np
import lightgbm as lgb
from scipy.stats import norm

# custom functions and constants
from IvsS_Utils import load_data, preprocess_data, split_data
from IvsS_Utils import nvps_profit
from IvsS_Utils import create_NN_model, tune_NN_model, compile_NN_model, train_NN_model


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
    path = "Data_Test_Multi_Raw\data.csv"
    multi_data = load_data(path, True)
    multi_feature_data, multi_target_data = preprocess_data(multi_data)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(multi_feature_data, multi_target_data)

    # Neural network - Complex
    best_estimator, hyperparameter, val_profit = tune_NN_model(X_train, y_train, X_val, y_val, alpha_data, underage_data, overage_data)
    model_ANN_complex = train_NN_model(hyperparameter, X_train, y_train, X_val, y_val, alpha_data, underage_data, overage_data)
    target_prediction_ANN = model_ANN_complex.predict(X_test)
    profit_complex_ANN = np.mean(nvps_profit(y_test, target_prediction_ANN, alpha_data, underage_data, overage_data))
