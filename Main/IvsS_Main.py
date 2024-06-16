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

# custom functions and constants
from IvsS_Utils import load_data, create_environment, load_dict, load_generated_data
from IvsS_Utils import ioa_ann_complex, ioa_ann_simple, ioa_xgb_complex, ioa_xgb_simple
from IvsS_Utils import soa_ann_complex, soa_ann_simple, soa_xgb_complex, soa_xgb_simple
from IvsS_Utils import ets_baseline



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
    path = "/pfs/work7/workspace/scratch/ma_elanza-thesislanza/"
    trials = 100

    # Load the dictionary for the datasets
    dataset_dict = load_dict(path=path)

    for i in range(len(dataset_dict)):
        print("Dataset: ", dataset_dict[i]['dataset_id'])
        print("Path: ", dataset_dict[i]['dataset_path'])
        print("Folder: ", dataset_dict[i]['folder_path'])
        run(path=dataset_dict[i]['dataset_path'], trials=trials, dataset_id=dataset_dict[i]['dataset_id'], save_path=dataset_dict[i]['folder_path'])
        

    
def run(path, trials, dataset_id, save_path):
    
    # Simple
    X_train, y_train, X_val, y_val, X_test, y_test = load_generated_data(path=path, multi=False)
    # Reshape y_train
    y_train = y_train.reshape(y_train.shape[0],1)
    y_val = y_val.reshape(y_val.shape[0],1)
    y_test = y_test.reshape(y_test.shape[0],1)
    print("X train shape", X_train.shape)
    print("y train shape", y_train.shape)
    
    try:
        ioa_ann_simple(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test, 
                   underage_data_single=underage_data_single, overage_data_single=overage_data_single, trials=trials, dataset_id=dataset_id, path=save_path)
    except Exception as e:
        print("Error in ioa_ann_simple", e)

    try:
        soa_ann_simple(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test, 
                   underage_data_single=underage_data_single, overage_data_single=overage_data_single, trials=trials, dataset_id=dataset_id, path=save_path)
    except Exception as e:
        print("Error in soa_ann_simple", e)

    try:
        ioa_xgb_simple(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test, 
                   underage_data_single=underage_data_single, overage_data_single=overage_data_single, trials=trials, dataset_id=dataset_id, path=save_path)
    except Exception as e:
        print("Error in ioa_xgb_simple", e)

    try: 
        soa_xgb_simple(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test, 
                   underage_data_single=underage_data_single, overage_data_single=overage_data_single, trials=trials, dataset_id=dataset_id, path=save_path)
    except Exception as e:  
        print("Error in soa_xgb_simple", e)
    
    # Complex  
    X_train, y_train, X_val, y_val, X_test, y_test = load_generated_data(path=path, multi=True)

    """
    print("start monte carlo")
    load_cost_structure(alpha_input=alpha_data, underage_input=underage_data, overage_input=overage_data)
    size = calculate_saa_size(y_train)
    print(size)
    """
    
    try:
        ioa_ann_complex(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test, 
                    alpha_data=alpha_data, underage_data=underage_data, overage_data=overage_data, trials=trials, dataset_id=dataset_id, path=save_path)
    except Exception as e:
        print("Error in ioa_ann_complex", e)

    try:
        soa_ann_complex(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test,
                    alpha_data=alpha_data, underage_data=underage_data, overage_data=overage_data, trials=trials, dataset_id=dataset_id, path=save_path)
    except Exception as e:
        print("Error in soa_ann_complex", e)

    try:
        ioa_xgb_complex(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test,
                    alpha_data=alpha_data, underage_data=underage_data, overage_data=overage_data, trials=trials, dataset_id=dataset_id, path=save_path)
    except Exception as e: 
        print("Error in ioa_xgb_complex", e)

    try:
        soa_xgb_complex(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test,
                    alpha_data=alpha_data, underage_data=underage_data, overage_data=overage_data, trials=trials, dataset_id=dataset_id, path=save_path)
    except Exception as e:
        print("Error in soa_xgb_complex", e)

    """
    try:
        ets_baseline(y_train=y_train, y_val=y_val, y_test=y_test, underage_data=underage_data, overage_data=overage_data, alpha_data=alpha_data, 
                    fit_past=27*7, dataset_id=dataset_id, path=save_path)
    except Exception as e:
        print("Error in ets_baseline", e)
    """


if __name__ == "__main__":
    
    create_environment()
    main()
    