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

#####################################################################################################################




def predict_SOA_ANN(X_train, X_test, target_train, target_test):
    # Define function to create the ANN model 
    def create_model(n_hidden, n_neurons, learning_rate, activation, input_shape=[15]): 
        model = Sequential()
        model.add(Dense(n_neurons, activation="relu", input_shape=input_shape))
        for layer in range(n_hidden):
            model.add(Dense(n_neurons, activation="relu"))
        model.add(Dense(1))
        optimizer = Adam() #learning_rate=learning_rate_input
        model.compile(loss="mean_squared_error", optimizer=optimizer)
        return model

    # Define function to build the model
    def model_builder(n_hidden=1, n_neurons=30, learning_rate=3e-3 , activation = 'relu'):
        return KerasRegressor(build_fn=create_model, verbose=0, n_hidden=n_hidden, n_neurons=n_neurons, learning_rate=learning_rate, activation=activation) 

    # Define the hyperparameters to search for the Randomized Search
    param_distribs = {
        "n_hidden": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "n_neurons": np.arange(1, 100),
        "learning_rate": reciprocal(1e-4, 1e-2),
        "batch_size": [16, 32, 64, 128],
        "epochs": [10, 20, 30, 40, 50],
        "activation": ['relu', 'sigmoid', 'tanh']
    }

    # Create the model and optimize with Randomized Search
    model_ANN = model_builder()
    rnd_search_cv_ANN = RandomizedSearchCV(model_ANN, param_distribs, n_iter=10, cv=3, scoring='neg_mean_squared_error')

    # Fit the model on the training data
    rnd_search_cv_ANN.fit(X_train, target_train)
    print(rnd_search_cv_ANN.best_params_)

    # Predict the target values
    target_pred_ANN = rnd_search_cv_ANN.predict(X_test)
    train_pred_ANN = rnd_search_cv_ANN.predict(X_train)

    mse = mean_squared_error(target_test, target_pred_ANN)
    print("MSE:", mse)

    return target_pred_ANN, train_pred_ANN


def predict_SOA_LGBM(X_train, X_test, target_train, target_test):
        # Create the model and define the hyperparameters to search for the Randomized Search
    model_DT = lgb.LGBMRegressor()
    param_distribs = {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.5],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }

    # Create the Randomized Search object
    rnd_search_cv_DT = RandomizedSearchCV(model_DT, param_distribs, n_iter=10, cv=3, scoring='neg_mean_squared_error')

    # Fit the model on the training data
    rnd_search_cv_DT.fit(X_train, target_train)

    # Get the best parameters and best estimator
    best_params = rnd_search_cv_DT.best_params_
    best_estimator = rnd_search_cv_DT.best_estimator_

    # Make predictions
    target_pred_DT = best_estimator.predict(X_test)
    train_pred_DT = best_estimator.predict(X_train)

    # Calculate the MSE
    mse = mean_squared_error(target_test, target_pred_DT)
    print("MSE:", mse)
    return target_pred_DT, train_pred_DT

def solve_SOA_non_parametric(target_pred_ANN, train_pred_ANN, target_pred_DT, train_pred_DT, target_test, target_train, q):
    # Initialize an empty list to store the final order quantities
    final_order_quantities_ANN = []
    final_order_quantities_DT = []

    # Calculate critical ratio
    critical_ratio = q

    scenario_size = 100

    # Initialize an empty list to store the final order quantities
    final_order_quantities_ANN = []

    # Calculate the forecast error for ANN on the prediction of the training data
    target_train_flatten = target_train.values.flatten()
    train_pred_ANN_flatten = train_pred_ANN.flatten()
    forecast_error_ANN = target_train_flatten - train_pred_ANN_flatten
    forecast_error_ANN_std = np.std(forecast_error_ANN)

    # Loop over each week in data_test
    for i in range(len(target_pred_ANN)):

        # Create Demand Scenarios for this week
        demand_scenarios = target_pred_ANN[i] + np.random.choice(forecast_error_ANN, size=scenario_size) 

        # Initialize a list to store the solutions for each scenario
        saa_solutions = []

        # For each demand scenario, solve the newsvendor problem
        for demand in demand_scenarios:
            
            # Calculate the solution for this scenario
            solution = norm.ppf(critical_ratio,loc=demand,scale=forecast_error_ANN_std)

            # Store the solution for this scenario
            saa_solutions.append(solution)

        # Average the solutions to get the final allocation
        final_allocation = np.mean(saa_solutions, axis=0)

        # Store the final order quantities
        final_order_quantities_ANN.append(final_allocation)

    # Initialize an empty list to store the final order quantities of DT
    final_order_quantities_DT = []

    # Calculate the forecast error for ANN on the prediction of the training data
    target_train_flatten = target_train.values.flatten()
    train_pred_DT_flatten = train_pred_DT.flatten()
    forecast_error_DT = target_train_flatten - train_pred_DT_flatten
    forecast_error_DT_std = np.std(forecast_error_DT)

    # Loop over each week in data_test
    for i in range(len(target_pred_DT)):

        # Create Demand Scenarios for this week
        demand_scenarios = target_pred_DT[i] + np.random.choice(forecast_error_DT, size=scenario_size) 

        # Initialize a list to store the solutions for each scenario
        saa_solutions = []

        # For each demand scenario, solve the newsvendor problem
        for demand in demand_scenarios:
            
            # Calculate the solution for this scenario
            solution = norm.ppf(critical_ratio,loc=demand,scale=forecast_error_DT_std)

            # Store the solution for this scenario
            saa_solutions.append(solution)

        # Average the solutions to get the final allocation
        final_allocation = np.mean(saa_solutions, axis=0)

        # Store the final order quantities
        final_order_quantities_DT.append(final_allocation)

    return final_order_quantities_ANN, final_order_quantities_DT

def solve_SOA_model_based(target_pred_ANN, train_pred_ANN, target_pred_DT, train_pred_DT, target_test, target_train, q):
    critical_ratio = q
    scenario_size = 100

    # Initialize an empty list to store the final order quantities
    final_order_quantities_ANN_model = []

    # Calculate the forecast error for ANN on the prediction of the training data
    target_train_flatten = target_train.values.flatten()
    train_pred_ANN_flatten = train_pred_ANN.flatten()
    forecast_error_ANN = target_train_flatten - train_pred_ANN_flatten
    forecast_error_ANN_std = np.std(forecast_error_ANN) 

    # Loop over each week in data_test
    for i in range(len(target_pred_ANN)):

        # Create Demand Scenarios for this week
        demand_scenarios = target_pred_ANN[i] + np.random.normal(loc=0, scale=forecast_error_ANN_std, size=scenario_size)

        # Initialize a list to store the solutions for each scenario
        saa_solutions = []

        # For each demand scenario, solve the newsvendor problem
        for demand in demand_scenarios:
            
            # Calculate the solution for this scenario
            solution = norm.ppf(critical_ratio,loc=demand,scale=forecast_error_ANN_std)

            # Store the solution for this scenario
            saa_solutions.append(solution)

        # Average the solutions to get the final allocation
        final_allocation = np.mean(saa_solutions, axis=0)

        # Store the final order quantities
        final_order_quantities_ANN_model.append(final_allocation)

    scenario_size = 100

    # Initialize an empty list to store the final order quantities
    final_order_quantities_DT_model = []

    # Calculate the forecast error for ANN on the prediction of the training data
    target_train_flatten = target_train.values.flatten()
    train_pred_DT_flatten = train_pred_DT.flatten()
    forecast_error_DT = target_train_flatten - train_pred_DT_flatten
    forecast_error_DT_std = np.std(forecast_error_DT)

    # Loop over each week in data_test
    for i in range(len(target_pred_DT)):

        # Create Demand Scenarios for this week
        demand_scenarios = target_pred_DT[i] + np.random.normal(loc=0, scale=forecast_error_DT_std, size=scenario_size)

        # Initialize a list to store the solutions for each scenario
        saa_solutions = []

        # For each demand scenario, solve the newsvendor problem
        for demand in demand_scenarios:
            
            # Calculate the solution for this scenario
            solution = norm.ppf(critical_ratio,loc=demand,scale=forecast_error_DT_std)

            # Store the solution for this scenario
            saa_solutions.append(solution)

        # Average the solutions to get the final allocation
        final_allocation = np.mean(saa_solutions, axis=0)

        # Store the final order quantities
        final_order_quantities_DT_model.append(final_allocation)

    return final_order_quantities_ANN_model, final_order_quantities_DT_model


def compare_results(final_order_quantities_ANN, final_order_quantities_DT, final_order_quantities_ANN_model, final_order_quantities_DT_model, target_test, p, c, s):
    # Loop over each week in target_test
    overall_costs_ANN = 0
    overall_costs_DT = 0
    overall_costs_ANN_model = 0
    overall_costs_DT_model = 0

    for i in range(len(target_test)):

        # Calculate understock and overstock costs
        cost_ANN = 0
        cost_DT = 0
        cost_ANN_model = 0
        cost_DT_model = 0

        # Costs for non-parametric
        if final_order_quantities_ANN[i] < target_test.values[i]:
            cost_ANN = (p - c) * (target_test.values[i] - np.round(final_order_quantities_ANN[i]))
        if final_order_quantities_ANN[i] > target_test.values[i]:
            cost_ANN = (c - s) * (np.round(final_order_quantities_ANN[i]) - target_test.values[i])
        if final_order_quantities_DT[i] < target_test.values[i]:
            cost_DT = (p - c) * (target_test.values[i] - np.round(final_order_quantities_DT[i]))
        if final_order_quantities_DT[i] > target_test.values[i]:
            cost_DT = (c - s) * (np.round(final_order_quantities_DT[i]) - target_test.values[i])
        # Costs for model-based
        if final_order_quantities_ANN_model[i] < target_test.values[i]:
            cost_ANN_model = (p - c) * (target_test.values[i] - np.round(final_order_quantities_ANN_model[i]))
        if final_order_quantities_ANN_model[i] > target_test.values[i]:
            cost_ANN_model = (c - s) * (np.round(final_order_quantities_ANN_model[i]) - target_test.values[i])
        if final_order_quantities_DT_model[i] < target_test.values[i]:
            cost_DT_model = (p - c) * (target_test.values[i] - np.round(final_order_quantities_DT_model[i]))
        if final_order_quantities_DT_model[i] > target_test.values[i]:
            cost_DT_model = (c - s) * (np.round(final_order_quantities_DT_model[i]) - target_test.values[i])

        # Calculate the total costs for the week
        overall_costs_ANN += cost_ANN
        overall_costs_DT += cost_DT
        overall_costs_ANN_model += cost_ANN_model
        overall_costs_DT_model += cost_DT_model
    

    # Print the overall profit
    print('Overall costs for ANN (non-parametric): ', int(overall_costs_ANN))
    print('Overall costs for DT (non-parametric): ', int(overall_costs_DT))
    print('Overall costs for ANN (model-based): ', int(overall_costs_ANN_model))
    print('Overall costs for DT (model-based): ', int(overall_costs_DT_model))

    return overall_costs_ANN, overall_costs_DT, overall_costs_ANN_model, overall_costs_DT_model


if __name__ == "__main__":
    print('Starte Programm...')

    # Define costs
    c = 0.3  # cost per unit of product
    s = 0.01  # salvage value per unit of leftover product
    p = 0.06  # price per unit of product sold

    # Calculate critical ratio
    critical_ratio = (p - c) / ((c - s) + (p - c))

    # Load Data
    X_train, X_test, target_train, target_test = load_data(multi=False)

    # Predict demand for simple Newsvendor with ANN
    target_pred_ANN, train_pred_ANN = predict_SOA_ANN(X_train, X_test, target_train, target_test)
    # Predict demand for simple Newsvendor with LGBM
    target_pred_DT, train_pred_DT = predict_SOA_LGBM(X_train, X_test, target_train, target_test)
    # Solve the simple Newsvendor problem non-parametric
    final_order_quantities_ANN, final_order_quantities_DT = solve_SOA_non_parametric(target_pred_ANN, train_pred_ANN, target_pred_DT, train_pred_DT, target_test, target_train, critical_ratio)
    # Solve the simple Newsvendor problem model-based
    final_order_quantities_ANN_model, final_order_quantities_DT_model = solve_SOA_model_based(target_pred_ANN, train_pred_ANN, target_pred_DT, train_pred_DT, target_test, target_train, critical_ratio)
    # Compare the results
    overall_costs_ANN, overall_costs_DT, overall_costs_ANN_model, overall_costs_DT_model = compare_results(final_order_quantities_ANN, final_order_quantities_DT, final_order_quantities_ANN_model, final_order_quantities_DT_model, target_test, p, c, s)

    # Print the overall profit
    print('Overall costs for ANN (non-parametric): ', int(overall_costs_ANN))
    print('Overall costs for DT (non-parametric): ', int(overall_costs_DT))
    print('Overall costs for ANN (model-based): ', int(overall_costs_ANN_model))
    print('Overall costs for DT (model-based): ', int(overall_costs_DT_model))

    print('Programm beendet.')






