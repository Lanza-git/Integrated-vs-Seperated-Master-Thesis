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
from IvsS_Utils import load_generated_data, preprocess_data, split_data,  create_environment
from IvsS_Utils import ets_forecast, ets_evaluate
from IvsS_Utils import tune_XGB_model, ets_baseline, load_dict
from IvsS_Utils import soa_ann_complex, soa_ann_simple, soa_xgb_complex, soa_xgb_simple
from IvsS_Utils import ioa_ann_complex, ioa_ann_simple, ioa_xgb_complex, ioa_xgb_simple


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

def main():
    from mpi4py import MPI
    import xgboost as xgb
    # Initialize the MPI communicator
    comm = MPI.COMM_WORLD


    # Get the rank of the current process
    rank = comm.Get_rank()

    trials = 100
    path = "/pfs/work7/workspace/scratch/ma_elanza-thesislanza/"
    # Load the dictionary for the datasets
    dataset_dict = load_dict(path=path)

    i = 3
    final_path =dataset_dict[i]['folder_path']
    path = dataset_dict[i]['dataset_path']
    dataset_id = dataset_dict[i]['dataset_id']

    # Define the message
    massage = str(rank)

    # Depending on the rank of the process, run a different approach
    if rank < 2:
        # Simple
        X_train, y_train, X_val, y_val, X_test, y_test = load_generated_data(path=path, multi=False)
        # Reshape y_train
        y_train = y_train.reshape(y_train.shape[0],1)
        y_val = y_val.reshape(y_val.shape[0],1)
        y_test = y_test.reshape(y_test.shape[0],1)
        print("X train shape", X_train.shape)
        print("y train shape", y_train.shape)

        if rank == 0: 
            # Integrated Optimization Approach - ANN - simple:
            ioa_ann_simple(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, 
                        y_test=y_test, underage_data_single=underage_data_single, overage_data_single=overage_data_single,
                        trials=trials, dataset_id=dataset_id, path=final_path)
            message = "ioa_ann_simple = Complete"

        elif rank == 1:
            # Seperate Optimization Approach - ANN - simple:
            soa_ann_simple(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, 
                        y_test=y_test, underage_data_single=underage_data_single, overage_data_single=overage_data_single,
                        trials=trials, dataset_id=dataset_id, path=final_path)
            message = "soa_ann_simple = Complete"

        elif rank == 2:
            # Integrated Optimization Approach - XGBoost - Simple:
            ioa_xgb_simple(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test,
                        y_test=y_test, underage_data_single=underage_data_single, overage_data_single=overage_data_single,
                        trials=trials, dataset_id=dataset_id, path=final_path)
            message = "ioa_xgb_simple = Complete"

        elif rank == 3:
            # Seperated Optimization Approach - XGBoost - Simple:
            soa_xgb_simple(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test,
                        y_test=y_test, underage_data_single=underage_data_single, overage_data_single=overage_data_single,
                        trials=trials, dataset_id=dataset_id, path=final_path)
            message = "soa_xgb_simple = Complete"

    elif rank >= 4:
        # Complex  
        X_train, y_train, X_val, y_val, X_test, y_test = load_generated_data(path=path, multi=True)
        print("X train shape", X_train.shape)
        print("y train shape", y_train.shape)

        if rank == 4:
            
            # ETS Forecasting:
            ets_baseline(y_train=y_train, y_val=y_val, y_test=y_test, underage_data=underage_data, overage_data=overage_data,
                        alpha_data=alpha_data, fit_past=10, dataset_id=dataset_id, path=final_path)
            message = "ets = complete"

        elif rank == 5:

            # Integrated Optimization Approach - ANN - complex:
            ioa_ann_complex(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test, 
                        alpha_data=alpha_data, underage_data=underage_data, overage_data=overage_data, 
                        trials=trials, dataset_id=dataset_id, path=final_path)
            message = "ioa_ann_complex = Complete"

        elif rank == 6:

            # Seperate Optimization Approach - ANN - complex:
            soa_ann_complex(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test, 
                        alpha_data=alpha_data, underage_data=underage_data, overage_data=overage_data, 
                        trials=trials, dataset_id=dataset_id, path=final_path)
            message = "soa_ann_complex = Complete"

        elif rank == 7:
            # Integrated Optimization Approach - XGBoost - Complex:
            ioa_xgb_complex(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test, 
                        alpha_data=alpha_data, underage_data=underage_data, overage_data=overage_data, 
                        trials=trials, dataset_id=dataset_id, path=final_path)
            message = "ioa_xgb_complex = Complete"

        elif rank == 8:
            # Seperated Optimization Approach - XGBoost - Complex:
            soa_xgb_complex(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test, 
                        alpha_data=alpha_data, underage_data=underage_data, overage_data=overage_data, 
                        trials=trials, dataset_id=dataset_id, path=final_path)
            message = "soa_xgb_complex = Complete"

    else:
        print("Invalid rank")

    # Define the message and the file path
    file_path = "/pfs/data5/home/ma/ma_ma/ma_elanza/test_dir/message_text.txt"

    # Open the file in append mode and write the message
    with open(file_path, 'a') as f:
        f.write(message + '\n')       

    comm.Barrier()

    print("finish")

if __name__ == "__main__":
    
    create_environment()

    main()

    import pickle



   