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

#load_packages()

# General imports
import numpy as np
import xgboost as xgb

# custom functions and constants
from IvsS_Utils import load_data, preprocess_data, split_data, nvps_profit, create_environment, ets_baseline
from IvsS_Utils import ioa_ann_complex, ioa_ann_simple, soa_ann_complex, soa_ann_simple
from IvsS_Utils import ioa_xgb_complex, ioa_xgb_simple, soa_xgb_complex, soa_xgb_simple


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



if __name__ == "__main__":
    create_environment()
    path = "Main/data.csv" 
    trials = 2
    dataset_id = "test_1"
    save_path = "test/"

    # Simple
    single_data = load_data(path=path, multi=False)
    single_feature_data, single_target_data = preprocess_data(raw_data=single_data)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(feature_data=single_feature_data, target_data=single_target_data)

    ioa_ann_simple(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test, 
                   underage_data_single=underage_data_single, overage_data_single=overage_data_single, trials=trials, dataset_id=dataset_id, path=save_path)
    
    soa_ann_simple(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test, 
                   underage_data_single=underage_data_single, overage_data_single=overage_data_single, trials=trials, dataset_id=dataset_id, path=save_path)

    ioa_xgb_simple(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test, 
                   underage_data_single=underage_data_single, overage_data_single=overage_data_single, trials=trials, dataset_id=dataset_id, path=save_path)
    
    soa_xgb_simple(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test, 
                   underage_data_single=underage_data_single, overage_data_single=overage_data_single, trials=trials, dataset_id=dataset_id, path=save_path)
    

    # Complex  
    multi_data = load_data(path=path, multi=True)
    multi_feature_data, multi_target_data = preprocess_data(raw_data=multi_data)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(feature_data=multi_feature_data, target_data=multi_target_data, test_size=0.2, val_size=0.2)
    
    ioa_ann_complex(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test, 
                    alpha_data=alpha_data, underage_data=underage_data, overage_data=overage_data, trials=trials, dataset_id=dataset_id, path=save_path)
    
    soa_ann_complex(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test,
                    alpha_data=alpha_data, underage_data=underage_data, overage_data=overage_data, trials=trials, dataset_id=dataset_id, path=save_path)
    
    ioa_xgb_complex(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test,
                    alpha_data=alpha_data, underage_data=underage_data, overage_data=overage_data, trials=trials, dataset_id=dataset_id, path=save_path)
    
    soa_xgb_complex(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test,
                    alpha_data=alpha_data, underage_data=underage_data, overage_data=overage_data, trials=trials, dataset_id=dataset_id, path=save_path)
    
    """
    ets_baseline(y_train=y_train, y_val=y_val, y_test=y_test, underage_data=underage_data, overage_data=overage_data, alpha_data=alpha_data, 
                    fit_past=27*7, dataset_id=dataset_id, path=save_path)
    """