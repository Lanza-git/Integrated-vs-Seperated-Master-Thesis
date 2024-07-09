# Load packages
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
    install('psutil')


load_packages()

# General imports
import numpy as np
import gc

# custom functions and constants
from IvsS_Utils import load_data, create_environment, load_dict, load_generated_data
from IvsS_Utils import ioa_ann_complex, ioa_ann_simple, ioa_xgb_complex, ioa_xgb_simple
from IvsS_Utils import soa_ann_complex, soa_ann_simple, soa_xgb_complex, soa_xgb_simple
from IvsS_Utils import ets_baseline



####################################### Constants ##############################################################################

def get_constants(risk_factor):
    # Parameters for multi-item newsvendor problem
    prices = np.array([0.3, 0.5, 0.6, 0.5, 0.5, 0.5]) #price data
    prices = prices.reshape(6,1)
    costs = np.array([0.06, 0.06, 0.06, 0.06, 0.06, 0.06]) #cost data
    costs = costs.reshape(6,1)
    costs = costs * risk_factor
    salvages = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01]) #salvage data
    salvages = salvages.reshape(6,1)
    underage_data = prices - costs 
    overage_data = costs - salvages 
    underage_data_single = underage_data[0,0]
    overage_data_single = overage_data[0,0]



    alpha_data = np.array([             #alpha data
        [0.0, 0.11, 0.15, 0.14, 0.14, 0.15],
        [0.22, 0.0, 0.08, 0.11, 0.12, 0.12],
        [0.24, 0.07, 0.0, 0.07, 0.08, 0.07],
        [0.26, 0.09, 0.07, 0.0, 0.12, 0.12],
        [0.19, 0.10, 0.10, 0.12, 0.0, 0.14],
        [0.17, 0.13, 0.11, 0.11, 0.13, 0.0]
    ])
    return underage_data, overage_data, alpha_data, underage_data_single, overage_data_single

####################################### Functions ##############################################################################

def main(dataset_id, risk_factor):

    # Path to the dataset
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

    if risk_factor != 1:
        folder_path = folder_path + "/risk_" + str(risk_factor*10)

    run(path=dataset_path, trials=trials, dataset_id=dataset_id, save_path=folder_path, risk_factor=risk_factor)

    
def run(path, trials, dataset_id, save_path, risk_factor=1):
    
    # Get constants
    underage_data, overage_data, alpha_data, underage_data_single, overage_data_single = get_constants(risk_factor)
    """
    # Simple
    X_train_single, y_train_single, X_val_single, y_val_single, X_test_single, y_test_single = load_generated_data(path=path, multi=False)
    # Reshape y_train
    y_train_single = y_train_single.reshape(y_train_single.shape[0],1)
    y_val_single = y_val_single.reshape(y_val_single.shape[0],1)
    y_test_single = y_test_single.reshape(y_test_single.shape[0],1)
    print("X train shape", X_train_single.shape)
    print("y train shape", y_train_single.shape)
    
    ioa_ann_simple(X_train=X_train_single, y_train=y_train_single, X_val=X_val_single, y_val=y_val_single, X_test=X_test_single, y_test=y_test_single, 
                underage_data_single=underage_data_single, overage_data_single=overage_data_single, trials=trials, dataset_id=dataset_id, path=save_path)
    print("ioa_ann_simple done: ", dataset_id)
    gc.collect()
    
    soa_ann_simple(X_train=X_train_single, y_train=y_train_single, X_val=X_val_single, y_val=y_val_single, X_test=X_test_single, y_test=y_test_single, 
                    underage_data_single=underage_data_single, overage_data_single=overage_data_single, trials=trials, dataset_id=dataset_id, path=save_path)
    print("soa_ann_simple done: ", dataset_id)
    gc.collect()
    
    ioa_xgb_simple(X_train=X_train_single, y_train=y_train_single, X_val=X_val_single, y_val=y_val_single, X_test=X_test_single, y_test=y_test_single, 
                underage_data_single=underage_data_single, overage_data_single=overage_data_single, trials=trials, dataset_id=dataset_id, path=save_path)
    print("ioa_xgb_simple done: ", dataset_id)
    gc.collect()
    
    soa_xgb_simple(X_train=X_train_single, y_train=y_train_single, X_val=X_val_single, y_val=y_val_single, X_test=X_test_single, y_test=y_test_single, 
                underage_data_single=underage_data_single, overage_data_single=overage_data_single, trials=trials, dataset_id=dataset_id, path=save_path)
    print("soa_xgb_simple done: ", dataset_id)
    gc.collect()

    # Drop unneeded variables
    del X_train_single, y_train_single, X_val_single, y_val_single, X_test_single, y_test_single
    
    """
    # Complex  
    X_train_multi, y_train_multi, X_val_multi, y_val_multi, X_test_multi, y_test_multi = load_generated_data(path=path, multi=True)
    print("X train shape", X_train_multi.shape)
    print("y train shape", y_train_multi.shape)

    
    ioa_ann_complex(X_train=X_train_multi, y_train=y_train_multi, X_val=X_val_multi, y_val=y_val_multi, X_test=X_test_multi, y_test=y_test_multi, 
                alpha_data=alpha_data, underage_data=underage_data, overage_data=overage_data, trials=trials, dataset_id=dataset_id, path=save_path)
    print("ioa_ann_complex done: ", dataset_id)
    gc.collect()
    
    try:
        soa_ann_complex(X_train=X_train_multi, y_train=y_train_multi, X_val=X_val_multi, y_val=y_val_multi, X_test=X_test_multi, y_test=y_test_multi,
                    alpha_data=alpha_data, underage_data=underage_data, overage_data=overage_data, trials=trials, dataset_id=dataset_id, path=save_path)
        print("soa_ann_complex done: ", dataset_id)
    except Exception as e:
        print("Error in soa_ann_complex", e)
    gc.collect()
    
    try:
        ioa_xgb_complex(X_train=X_train_multi, y_train=y_train_multi, X_val=X_val_multi, y_val=y_val_multi, X_test=X_test_multi, y_test=y_test_multi,
                    alpha_data=alpha_data, underage_data=underage_data, overage_data=overage_data, trials=trials, dataset_id=dataset_id, path=save_path)
        print("ioa_xgb_complex done: ", dataset_id)
    except Exception as e: 
        print("Error in ioa_xgb_complex", e)
    gc.collect()
    
    try:
        soa_xgb_complex(X_train=X_train_multi, y_train=y_train_multi, X_val=X_val_multi, y_val=y_val_multi, X_test=X_test_multi, y_test=y_test_multi,
                    alpha_data=alpha_data, underage_data=underage_data, overage_data=overage_data, trials=trials, dataset_id=dataset_id, path=save_path)
        print("soa_xgb_complex done: ", dataset_id)
    except Exception as e:
        print("Error in soa_xgb_complex", e)
    gc.collect()
    
    """
    try:
        ets_baseline(y_train=y_train, y_val=y_val, y_test=y_test, underage_data=underage_data, overage_data=overage_data, alpha_data=alpha_data, 
                    fit_past=27*7, dataset_id=dataset_id, path=save_path)
        print("ets_baseline done: ", dataset_id)
    except Exception as e:
        print("Error in ets_baseline", e)
    """

    # Drop unneeded variables
    del X_train_multi, y_train_multi, X_val_multi, y_val_multi, X_test_multi, y_test_multi


if __name__ == "__main__":
    import sys

    # Check if the dataset_id is passed to the script
    if len(sys.argv) > 1:
        dataset_id = sys.argv[1]
        risk_factor = float(sys.argv[2])
    else:
        print("No dataset_id provided")
        sys.exit(1)  # Exit the script if no dataset_id is provided

    create_environment()
    main(dataset_id, risk_factor)
