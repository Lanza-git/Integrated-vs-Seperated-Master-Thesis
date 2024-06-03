# Load packages
#from IvsS_Utils import load_packages




# General imports
import numpy as np

# custom functions and constants
from IvsS_Utils import load_data, preprocess_data, split_data, nvps_profit, solve_MILP, create_environment
from IvsS_Utils import train_NN_model, tune_NN_model_optuna, solve_basic_newsvendor_seperate, solve_complex_newsvendor_seperate
from IvsS_Utils import ets_forecast, ets_evaluate


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

def main(path):
    from mpi4py import MPI
    import pickle
    # Initialize the MPI communicator
    comm = MPI.COMM_WORLD

    # Get the rank of the current process
    rank = comm.Get_rank()

    # Depending on the rank of the process, run a different approach
    if rank < 2:

        # Load single Data
        single_data = load_data(path, False)
        single_feature_data, single_target_data = preprocess_data(single_data)
        X_train, y_train, X_val, y_val, X_test, y_test = split_data(single_feature_data, single_target_data)

        if rank == 0: 

            # Integrated Optimization Approach:
            best_estimator, hyperparameter, val_profit = tune_NN_model_optuna(X_train, y_train, X_val, y_val, None, underage_data_single, overage_data_single, multi = False)
            model_ANN_simple = train_NN_model(hyperparameter, X_train, y_train, X_val, y_val, None, underage_data_single, overage_data_single, multi = False)
            target_prediction_ANN = model_ANN_simple.predict(X_test)
            profit_simple_ANN_IOA = np.mean(nvps_profit(y_test, target_prediction_ANN, None, underage_data_single, overage_data_single))
            results = {'profit_simple_ANN_IOA': profit_simple_ANN_IOA}
        
        if rank == 1:

            # Seperate Optimization Approach:
            best_estimator, hyperparameter, val_profit = tune_NN_model_optuna(X_train, y_train, X_val, y_val, alpha_data, underage_data, overage_data, multi = True, integrated = False)
            model_ANN_complex = train_NN_model(hyperparameter, X_train, y_train, X_val, y_val, alpha_data, underage_data, overage_data, multi = True,  integrated = False)
            target_prediction_ANN = model_ANN_complex.predict(X_test)
            train_prediction_ANN = model_ANN_complex.predict(X_train)
            orders_scp_ann, orders_scnp_ann = solve_complex_newsvendor_seperate(y_train=y_train, y_train_pred=train_prediction_ANN, y_test_pred=target_prediction_ANN, u=underage_data, o=overage_data, alpha=alpha_data)
            profit_scp_ANN = np.mean(nvps_profit(y_test, orders_scp_ann, alpha_data, underage_data, overage_data))
            profit_scnp_ANN = np.mean(nvps_profit(y_test, orders_scnp_ann, alpha_data, underage_data, overage_data))
            results = {'profit_scp_ANN': profit_scp_ANN, 'profit_scnp_ANN': profit_scnp_ANN}

    elif rank >= 2:

        multi_data = load_data(path, True)
        multi_feature_data, multi_target_data = preprocess_data(multi_data)
        X_train, y_train, X_val, y_val, X_test, y_test = split_data(multi_feature_data, multi_target_data)

        if rank == 2:

            results_dct, elapse_time = ets_forecast(y_train=y_train, y_val=y_val, y_test_length=y_test.shape[0], fit_past=10)
            profit_single_ets, profit_multi_ets = ets_evaluate(y_test, results_dct, underage_data, overage_data, alpha_data)
            results = {'profit_single_ets': profit_single_ets, 'profit_multi_ets': profit_multi_ets, 'elapsed_time_ets': elapse_time}

        if rank == 3:

            # Integrated Optimization Approach:
            best_estimator, hyperparameter, val_profit = tune_NN_model_optuna(X_train, y_train, X_val, y_val, alpha_data, underage_data, overage_data)
            model_ANN_complex = train_NN_model(hyperparameter, X_train, y_train, X_val, y_val, alpha_data, underage_data, overage_data)
            target_prediction_ANN = model_ANN_complex.predict(X_test)
            profit_complex_ANN_IOA = np.mean(nvps_profit(y_test, target_prediction_ANN, alpha_data, underage_data, overage_data))
            results = {'profit_complex_ANN_IOA': profit_complex_ANN_IOA}

        if rank == 4:

            # Seperate Optimization Approach:
            best_estimator, hyperparameter, val_profit = tune_NN_model_optuna(X_train, y_train, X_val, y_val, alpha_data, underage_data, overage_data, multi = True, integrated = False)
            model_ANN_complex = train_NN_model(hyperparameter, X_train, y_train, X_val, y_val, alpha_data, underage_data, overage_data, multi = True,  integrated = False)
            target_prediction_ANN = model_ANN_complex.predict(X_test)
            train_prediction_ANN = model_ANN_complex.predict(X_train)
            orders_scp_ann, orders_scnp_ann = solve_complex_newsvendor_seperate(y_train=y_train, y_train_pred=train_prediction_ANN, y_test_pred=target_prediction_ANN, u=underage_data, o=overage_data, alpha=alpha_data)
            profit_scp_ANN = np.mean(nvps_profit(y_test, orders_scp_ann, alpha_data, underage_data, overage_data))
            profit_scnp_ANN = np.mean(nvps_profit(y_test, orders_scnp_ann, alpha_data, underage_data, overage_data))
            results = {'profit_scp_ANN': profit_scp_ANN, 'profit_scnp_ANN': profit_scnp_ANN}

    else:
        print("Invalid rank")

    # Save the results to a pickle file
    with open('pfs/data5/home/ma/ma_ma/ma_elanza/test_dir/results_rank_{}.pkl'.format(rank), 'wb') as f:
        pickle.dump(results, f)

if __name__ == "__main__":
    
    load_packages()
    create_environment()

    path = "/pfs/data5/home/ma/ma_ma/ma_elanza/test_dir/data.csv"

    main(path)

    import pickle

    # Load the results from each pickle file
    with open('results_rank_2.pkl', 'rb') as f:
        results_2 = pickle.load(f)
    with open('results_rank_3.pkl', 'rb') as f:
        results_3 = pickle.load(f)
    with open('results_rank_4.pkl', 'rb') as f:
        results_4 = pickle.load(f)

    # Combine the results into one dictionary
    combined_results = {**results_2, **results_3, **results_4}

    print("Combined Results: \n", combined_results)

    # Save the combined results to a new pickle file
    with open('combined_results.pkl', 'wb') as f:
        pickle.dump(combined_results, f)

   