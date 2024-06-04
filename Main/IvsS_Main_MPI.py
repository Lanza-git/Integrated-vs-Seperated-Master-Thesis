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
    install('tensorflow')#<2.13
    install('mpi4py')

load_packages()


# General imports
import numpy as np

# custom functions and constants
from IvsS_Utils import load_data, preprocess_data, split_data, nvps_profit, solve_MILP, create_environment
from IvsS_Utils import train_NN_model, tune_NN_model_optuna, solve_basic_newsvendor_seperate, solve_complex_newsvendor_seperate
from IvsS_Utils import ets_forecast, ets_evaluate
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

def main(path):
    from mpi4py import MPI
    import pickle
    import xgboost as xgb
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
            # Integrated Optimization Approach - ANN - simple:
            best_estimator, hyperparameter, val_profit = tune_NN_model_optuna(X_train, y_train, X_val, y_val, None, underage_data_single, overage_data_single, multi = False)
            model_ANN_simple = train_NN_model(hyperparameter, X_train, y_train, X_val, y_val, None, underage_data_single, overage_data_single, multi = False)
            target_prediction_ANN = model_ANN_simple.predict(X_test)
            profit_simple_ANN_IOA = np.mean(nvps_profit(y_test, target_prediction_ANN, None, underage_data_single, overage_data_single))
            results = {'profit_simple_ANN_IOA': profit_simple_ANN_IOA}

        elif rank == 1:
            # Seperate Optimization Approach - ANN - simple:
            best_estimator, hyperparameter, val_profit = tune_NN_model_optuna(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, alpha_input=None, underage_input=underage_data_single, overage_input=overage_data_single, multi = False, integrated = False, trials=trials)   
            model_ANN_simple = train_NN_model(hp=hyperparameter, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, alpha=None, underage=underage_data_single, overage=overage_data_single, multi = False, integrated = False)
            target_prediction_ANN = model_ANN_simple.predict(X_test)
            train_prediction_ANN = model_ANN_simple.predict(X_train)
            orders_ssp_ann, orders_ssnp_ann = solve_basic_newsvendor_seperate(y_train=y_train, y_train_pred=train_prediction_ANN, y_test_pred=target_prediction_ANN, u=underage_data_single, o=overage_data_single)
            profit_ssp_ANN = np.mean(nvps_profit(demand=y_test, q=orders_ssp_ann, alpha=None, u=underage_data_single, o=overage_data_single))
            profit_ssnp_ANN = np.mean(nvps_profit(demand=y_test, q=orders_ssnp_ann, alpha=None, u=underage_data_single, o=overage_data_single))
            results = {'profit_scp_ANN': profit_scp_ANN, 'profit_scnp_ANN': profit_scnp_ANN}

        elif rank == 2:
            # Integrated Optimization Approach - XGBoost - Simple:
            xgb_model, params, results = tune_XGB_model(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, alpha_input=None, underage_input=underage_data_single, overage_input=overage_data_single, multi=False, integrated=True)
            xgb_result = xgb_model.predict(xgb.DMatrix(X_test))
            profit_simple_XGB_IOA = np.mean(nvps_profit(demand=y_test, q=xgb_result, alpha=None, u=underage_data_single, o=overage_data_single))
            results = {'profit_simple_XGB_IOA': profit_simple_XGB_IOA}

        elif rank == 3:
            # Seperated Optimization Approach - XGBoost - Simple:
            xgb_model, hyperparameter_XGB_SOA_Complex, val_profit = tune_XGB_model(X_train, y_train, X_val, y_val, None, underage_data_single, overage_data_single, multi = False, integrated = False)
            target_prediction_XGB = xgb_model.predict(xgb.DMatrix(X_test))
            train_prediction_XGB = xgb_model.predict(xgb.DMatrix(X_train))  
            orders_ssp_XGB, orders_ssnp_XGB = solve_basic_newsvendor_seperate(y_train=y_train, y_train_pred=train_prediction_XGB, y_test_pred=target_prediction_XGB, u=underage_data_single, o=overage_data_single)
            profit_ssp_XGB = np.mean(nvps_profit(y_test, orders_ssp_XGB, alpha_data, underage_data_single, overage_data_single))
            profit_ssnp_XGB = np.mean(nvps_profit(y_test, orders_ssnp_XGB, alpha_data, underage_data_single, overage_data_single))

    elif rank >= 4:
        # Load multi Data
        multi_data = load_data(path, True)
        multi_feature_data, multi_target_data = preprocess_data(multi_data)
        X_train, y_train, X_val, y_val, X_test, y_test = split_data(multi_feature_data, multi_target_data)

        if rank == 4:
            
            # ETS Forecasting:
            results_dct, elapse_time = ets_forecast(y_train=y_train, y_val=y_val, y_test_length=y_test.shape[0], fit_past=10)
            profit_single_ets, profit_multi_ets = ets_evaluate(y_test, results_dct, underage_data, overage_data, alpha_data)
            results = {'profit_single_ets': profit_single_ets, 'profit_multi_ets': profit_multi_ets, 'elapsed_time_ets': elapse_time}

            with open('/pfs/data5/home/ma/ma_ma/ma_elanza/test_dir/message_text.txt', 'a') as f:
                f.write("2 complete" + '\n')

        elif rank == 5:

            # Integrated Optimization Approach - ANN - complex:
            best_estimator, hyperparameter, val_profit = tune_NN_model_optuna(X_train, y_train, X_val, y_val, alpha_data, underage_data, overage_data)
            model_ANN_complex = train_NN_model(hyperparameter, X_train, y_train, X_val, y_val, alpha_data, underage_data, overage_data)
            target_prediction_ANN = model_ANN_complex.predict(X_test)
            profit_complex_ANN_IOA = np.mean(nvps_profit(y_test, target_prediction_ANN, alpha_data, underage_data, overage_data))
            results = {'profit_complex_ANN_IOA': profit_complex_ANN_IOA}

            with open('/pfs/data5/home/ma/ma_ma/ma_elanza/test_dir/message_text.txt', 'a') as f:
                f.write("3 complete" + '\n')

        elif rank == 6:

            # Seperate Optimization Approach - ANN - complex:
            best_estimator, hyperparameter, val_profit = tune_NN_model_optuna(X_train, y_train, X_val, y_val, alpha_data, underage_data, overage_data, multi = True, integrated = False)
            model_ANN_complex = train_NN_model(hyperparameter, X_train, y_train, X_val, y_val, alpha_data, underage_data, overage_data, multi = True,  integrated = False)
            target_prediction_ANN = model_ANN_complex.predict(X_test)
            train_prediction_ANN = model_ANN_complex.predict(X_train)
            orders_scp_ann, orders_scnp_ann = solve_complex_newsvendor_seperate(y_train=y_train, y_train_pred=train_prediction_ANN, y_test_pred=target_prediction_ANN, u=underage_data, o=overage_data, alpha=alpha_data)
            profit_scp_ANN = np.mean(nvps_profit(y_test, orders_scp_ann, alpha_data, underage_data, overage_data))
            profit_scnp_ANN = np.mean(nvps_profit(y_test, orders_scnp_ann, alpha_data, underage_data, overage_data))
            results = {'profit_scp_ANN': profit_scp_ANN, 'profit_scnp_ANN': profit_scnp_ANN}
            
            with open('/pfs/data5/home/ma/ma_ma/ma_elanza/test_dir/message_text.txt', 'a') as f:
                f.write("4 complete" + '\n')

        elif rank == 7:
            # Integrated Optimization Approach - XGBoost - Complex:
            xgb_model, params, results = tune_XGB_model(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, alpha_input=alpha_data, underage_input=underage_data, overage_input=overage_data, integrated=True, trials=trials)
            xgb_result = xgb_model.predict(xgb.DMatrix(X_test))
            profit_complex_XGB_IOA = np.mean(nvps_profit(demand=y_test, q=xgb_result, alpha=alpha_data, u=underage_data, o=overage_data))
            results = {'profit_complex_XGB_IOA': profit_complex_XGB_IOA}

        elif rank == 8:
            # Seperated Optimization Approach - XGBoost - Complex:
            xgb_model, hyperparameter_XGB_SOA_Complex, val_profit = tune_XGB_model(X_train, y_train, X_val, y_val, alpha_data, underage_data, overage_data, multi = True, integrated = False)
            target_prediction_XGB = xgb_model.predict(xgb.DMatrix(X_test))
            train_prediction_XGB = xgb_model.predict(xgb.DMatrix(X_train))
            orders_scp_XGB, orders_scnp_XGB = solve_complex_newsvendor_seperate(y_train=y_train, y_train_pred=train_prediction_XGB, y_test_pred=target_prediction_XGB, u=underage_data, o=overage_data, alpha=alpha_data)
            profit_scp_XGB = np.mean(nvps_profit(y_test, orders_scp_XGB, alpha_data, underage_data, overage_data))
            profit_scnp_XGB = np.mean(nvps_profit(y_test, orders_scnp_XGB, alpha_data, underage_data, overage_data))
            results = {'profit_scp_XGB': profit_scp_XGB, 'profit_scnp_XGB': profit_scnp_XGB}

    else:
        print("Invalid rank")

    # Save the results to a pickle file
    with open('/pfs/data5/home/ma/ma_ma/ma_elanza/test_dir/Results_test/results_rank_{}.pkl'.format(rank), 'wb') as f:
        pickle.dump(results, f)

    comm.Barrier()

    # Initialize an empty dictionary for the combined results
    combined_results = {}

    # Load the results from each pickle file
    for i in range(9):
        with open(f'/pfs/data5/home/ma/ma_ma/ma_elanza/test_dir/Results_test/results_rank_{i}.pkl', 'rb') as f:
            results = pickle.load(f)
        # Combine the results into the dictionary
        combined_results.update(results)

    print("Combined Results: \n", combined_results)

    import json

    # Save the combined results to a new text file
    with open('combined_results.txt', 'w') as f:
        f.write(json.dumps(combined_results, indent=4))

if __name__ == "__main__":
    
    create_environment()

    path = "/pfs/data5/home/ma/ma_ma/ma_elanza/test_dir/data.csv"

    main(path)

    import pickle



   