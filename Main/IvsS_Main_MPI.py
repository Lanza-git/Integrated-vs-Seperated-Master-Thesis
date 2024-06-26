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

def main(dataset_id):



    path =  "/pfs/work7/workspace/scratch/ma_elanza-thesislanza/"
    trials = 100
    
    # Load the dictionary for the datasets
    dataset_dict = load_dict(path=path)

    # Get thr right entry from the dictionary
    for dataset in dataset_dict:
        if dataset['dataset_id'] == dataset_id:
            dataset_path = dataset['dataset_path']
            folder_path = dataset['folder_path']
            break

    run(path=dataset_path, trials=trials, dataset_id=dataset_id, final_path=folder_path)



def run(path, trials, dataset_id, final_path):
    from mpi4py import MPI

    # Initialize the MPI communicator
    comm = MPI.COMM_WORLD
    # Get the rank of the current process
    rank = comm.Get_rank()
    # Define the message
    massage = str(rank)

    # Depending on the rank of the process, run a different approach
    if rank < 2:
    
        # Simple
        X_train_single, y_train_single, X_val_single, y_val_single, X_test_single, y_test_single = load_generated_data(path=path, multi=False)
        # Reshape y_train
        y_train_single = y_train_single.reshape(y_train_single.shape[0],1)
        y_val_single = y_val_single.reshape(y_val_single.shape[0],1)
        y_test_single = y_test_single.reshape(y_test_single.shape[0],1)
        print("X train shape", X_train_single.shape)
        print("y train shape", y_train_single.shape)

        if rank == 0: 
            # Integrated Optimization Approach - ANN - simple:
            ioa_ann_simple(X_train=X_train_single, y_train=y_train_single, X_val=X_val_single, y_val=y_val_single, X_test=X_test_single, 
                        y_test=y_test_single, underage_data_single=underage_data_single, overage_data_single=overage_data_single,
                        trials=trials, dataset_id=dataset_id, path=final_path)
            message = "ioa_ann_simple = Complete " + str(dataset_id)

        elif rank == 1:
            # Seperate Optimization Approach - ANN - simple:
            soa_ann_simple(X_train=X_train_single, y_train=y_train_single, X_val=X_val_single, y_val=y_val_single, X_test=X_test_single, 
                        y_test=y_test_single, underage_data_single=underage_data_single, overage_data_single=overage_data_single,
                        trials=trials, dataset_id=dataset_id, path=final_path)
            message = "soa_ann_simple = Complete " + str(dataset_id)

        elif rank == 2:
            # Integrated Optimization Approach - XGBoost - Simple:
            ioa_xgb_simple(X_train=X_train_single, y_train=y_train_single, X_val=X_val_single, y_val=y_val_single, X_test=X_test_single,
                        y_test=y_test_single, underage_data_single=underage_data_single, overage_data_single=overage_data_single,
                        trials=trials, dataset_id=dataset_id, path=final_path)
            message = "ioa_xgb_simple = Complete " + str(dataset_id)

        elif rank == 3:
            # Seperated Optimization Approach - XGBoost - Simple:
            soa_xgb_simple(X_train=X_train_single, y_train=y_train_single, X_val=X_val_single, y_val=y_val_single, X_test=X_test_single,
                        y_test=y_test_single, underage_data_single=underage_data_single, overage_data_single=overage_data_single,
                        trials=trials, dataset_id=dataset_id, path=final_path)
            message = "soa_xgb_simple = Complete " + str(dataset_id)

        del X_train_single, y_train_single, X_val_single, y_val_single, X_test_single, y_test_single

    elif rank >= 4:
        
        # Complex  
        X_train_multi, y_train_multi, X_val_multi, y_val_multi, X_test_multi, y_test_multi = load_generated_data(path=path, multi=True)
        print("X train shape", X_train_multi.shape)
        print("y train shape", y_train_multi.shape)

        if rank == 4:
            
            # ETS Forecasting:
            ets_baseline(y_train=y_train_multi, y_val=y_val_multi, y_test=y_test_multi, underage_data=underage_data, overage_data=overage_data,
                        alpha_data=alpha_data, fit_past=10, dataset_id=dataset_id, path=final_path)
            message = "ets = complete " + str(dataset_id)

        elif rank == 5:

            # Integrated Optimization Approach - ANN - complex:
            ioa_ann_complex(X_train=X_train_multi, y_train=y_train_multi, X_val=X_val_multi, y_val=y_val_multi, X_test=X_test_multi, y_test=y_test_multi, 
                        alpha_data=alpha_data, underage_data=underage_data, overage_data=overage_data, 
                        trials=trials, dataset_id=dataset_id, path=final_path)
            message = "ioa_ann_complex = Complete " + str(dataset_id)

        elif rank == 6:

            # Seperate Optimization Approach - ANN - complex:
            soa_ann_complex(X_train=X_train_multi, y_train=y_train_multi, X_val=X_val_multi, y_val=y_val_multi, X_test=X_test_multi, y_test=y_test_multi, 
                        alpha_data=alpha_data, underage_data=underage_data, overage_data=overage_data, 
                        trials=trials, dataset_id=dataset_id, path=final_path)
            message = "soa_ann_complex = Complete " + str(dataset_id)

        elif rank == 7:
            # Integrated Optimization Approach - XGBoost - Complex:
            ioa_xgb_complex(X_train=X_train_multi, y_train=y_train_multi, X_val=X_val_multi, y_val=y_val_multi, X_test=X_test_multi, y_test=y_test_multi, 
                        alpha_data=alpha_data, underage_data=underage_data, overage_data=overage_data, 
                        trials=trials, dataset_id=dataset_id, path=final_path)
            message = "ioa_xgb_complex = Complete " + str(dataset_id)

        elif rank == 8:
            # Seperated Optimization Approach - XGBoost - Complex:
            soa_xgb_complex(X_train=X_train_multi, y_train=y_train_multi, X_val=X_val_multi, y_val=y_val_multi, X_test=X_test_multi, y_test=y_test_multi, 
                        alpha_data=alpha_data, underage_data=underage_data, overage_data=overage_data, 
                        trials=trials, dataset_id=dataset_id, path=final_path)
            message = "soa_xgb_complex = Complete " + str(dataset_id)

        del X_train_multi, y_train_multi, X_val_multi, y_val_multi, X_test_multi, y_test_multi

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

    import sys

    # Check if the dataset_id is passed to the script
    if len(sys.argv) > 1:
        dataset_id = sys.argv[1]
    else:
        print("No dataset_id provided")
        sys.exit(1)  # Exit the script if no dataset_id is provided

     
    create_environment()
    main(dataset_id)




   