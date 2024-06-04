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
import xgboost as xgb

# custom functions and constants
from IvsS_Utils import load_data, preprocess_data, split_data, nvps_profit, create_environment
from IvsS_Utils import train_NN_model, tune_NN_model_optuna
from IvsS_Utils import solve_basic_newsvendor_seperate, solve_complex_newsvendor_seperate
from IvsS_Utils import tune_XGB_model
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



if __name__ == "__main__":
    
    #load_packages()
    create_environment()

    path = "/pfs/data5/home/ma/ma_ma/ma_elanza/test_dir/data.csv" # "Main/data.csv" 

    trials = 10


    """
    TODOS
    - IOA complex not working - resolved
    - SOA is cheating because of optuna - Resolved: check with Xander
    - XGBoost - resolved
    - Baseline - ARIMA 
    - Baseline - ETS - resolved
    - seperate optimization approach - the gaussian process is not working (1 value for all products)

    
       
    """
    
    # Neural network - Complex  
    multi_data = load_data(path=path, multi=True)
    multi_feature_data, multi_target_data = preprocess_data(raw_data=multi_data)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(feature_data=multi_feature_data, target_data=multi_target_data, test_size=0.2, val_size=0.2)
    
    
    # Baseline - ETS
    results_dct, elapse_time = ets_forecast(y_train=y_train, y_val=y_val, y_test_length=y_test.shape[0], fit_past=10)
    profit_single_ets, profit_multi_ets = ets_evaluate(y_test=y_test, results_dct=results_dct, underage=underage_data, overage=overage_data, alpha=alpha_data)

    print("Step 0")
    
    # Integrated Optimization Approach - Neural Network - Complex:
    best_estimator, hyperparameter, val_profit = tune_NN_model_optuna(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, alpha_input=alpha_data, underage_input=underage_data, overage_input=overage_data, trials=trials)
    model_ANN_complex = train_NN_model(hp=hyperparameter, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, alpha=alpha_data, underage=underage_data, overage=overage_data)
    target_prediction_ANN = model_ANN_complex.predict(X_test)
    profit_complex_ANN_IOA = np.mean(nvps_profit(demand=y_test, q=target_prediction_ANN, alpha=alpha_data, u=underage_data, o=overage_data))
    

    print("Step 1")
    
    # Seperate Optimization Approach - Neural Network - Complex:
    best_estimator, hyperparameter, val_profit = tune_NN_model_optuna(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, alpha_input=alpha_data, underage_input=underage_data, overage_input=overage_data, multi = True, integrated = False, trials=trials)
    model_ANN_complex = train_NN_model(hp=hyperparameter, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, alpha=alpha_data, underage=underage_data, overage=overage_data, multi = True,  integrated = False)
    target_prediction_ANN = model_ANN_complex.predict(X_test)
    train_prediction_ANN = model_ANN_complex.predict(X_train)
    print("ann prediction - complete")
    orders_scp_ann, orders_scnp_ann = solve_complex_newsvendor_seperate(y_train=y_train, y_train_pred=train_prediction_ANN, y_test_pred=target_prediction_ANN, u=underage_data, o=overage_data, alpha=alpha_data)
    print("order allocation - complete")
    profit_scp_ANN = np.mean(nvps_profit(demand=y_test, q=orders_scp_ann, alpha=alpha_data, u=underage_data, o=overage_data))
    profit_scnp_ANN = np.mean(nvps_profit(demand=y_test, q=orders_scnp_ann, alpha=alpha_data, u=underage_data, o=overage_data))
    
    print("Step 2")
    
    # Integrated Optimization Approach - XGBoost - Complex:
    xgb_model, params, results = tune_XGB_model(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, alpha_input=alpha_data, underage_input=underage_data, overage_input=overage_data, integrated=True, trials=trials)
    xgb_result = xgb_model.predict(xgb.DMatrix(X_test))
    profit_complex_XGB_IOA = np.mean(nvps_profit(demand=y_test, q=xgb_result, alpha=alpha_data, u=underage_data, o=overage_data))
    
    print("profit complex", profit_complex_XGB_IOA)
    
    # Seperated Optimization Approach - XGBoost - Complex:
    xgb_model, hyperparameter_XGB_SOA_Complex, val_profit = tune_XGB_model(X_train, y_train, X_val, y_val, alpha_data, underage_data, overage_data, multi = True, integrated = False)
    target_prediction_XGB = xgb_model.predict(xgb.DMatrix(X_test))
    train_prediction_XGB = xgb_model.predict(xgb.DMatrix(X_train))
    orders_scp_XGB, orders_scnp_XGB = solve_complex_newsvendor_seperate(y_train=y_train, y_train_pred=train_prediction_XGB, y_test_pred=target_prediction_XGB, u=underage_data, o=overage_data, alpha=alpha_data)
    profit_scp_XGB = np.mean(nvps_profit(y_test, orders_scp_XGB, alpha_data, underage_data, overage_data))
    profit_scnp_XGB = np.mean(nvps_profit(y_test, orders_scnp_XGB, alpha_data, underage_data, overage_data))

    # Optimal profit
    optimal_profit_multi = np.mean(nvps_profit(demand=y_test, q=y_test, alpha=alpha_data, u=underage_data, o=overage_data))
    
    print("Step 2: "+ str(profit_scnp_XGB), profit_scp_XGB)
    
    # Neural network - Simple
    single_data = load_data(path=path, multi=False)
    single_feature_data, single_target_data = preprocess_data(raw_data=single_data)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(feature_data=single_feature_data, target_data=single_target_data)
    
    print("Simple Data loaded")

    # Integrated Optimization Approach - Neural Network - Simple:
    best_estimator, hyperparameter, val_profit = tune_NN_model_optuna(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, alpha_input=alpha_data, underage_input=underage_data_single, overage_input=overage_data_single, multi = False, trials=trials)
    print("nn model - complete - simple")
    model_ANN_simple = train_NN_model(hp=hyperparameter, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, alpha=alpha_data, underage=underage_data_single, overage=overage_data_single, multi = False)
    print("model trained - simple")
    target_prediction_ANN = model_ANN_simple.predict(X_test)
    print("prediction - complete")
    profit_simple_ANN_IOA = np.mean(nvps_profit(demand=y_test, q=target_prediction_ANN, alpha=alpha_data, u=underage_data_single, o=overage_data_single))
    
    print("Step 3")

    # Seperate Optimization Approach - Neural Network - Simple:
    best_estimator, hyperparameter, val_profit = tune_NN_model_optuna(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, alpha_input=None, underage_input=underage_data_single, overage_input=overage_data_single, multi = False, integrated = False, trials=trials)   
    model_ANN_simple = train_NN_model(hp=hyperparameter, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, alpha=None, underage=underage_data_single, overage=overage_data_single, multi = False, integrated = False)
    target_prediction_ANN = model_ANN_simple.predict(X_test)
    train_prediction_ANN = model_ANN_simple.predict(X_train)
    orders_ssp_ann, orders_ssnp_ann = solve_basic_newsvendor_seperate(y_train=y_train, y_train_pred=train_prediction_ANN, y_test_pred=target_prediction_ANN, u=underage_data_single, o=overage_data_single)
    profit_ssp_ANN = np.mean(nvps_profit(demand=y_test, q=orders_ssp_ann, alpha=None, u=underage_data_single, o=overage_data_single))
    profit_ssnp_ANN = np.mean(nvps_profit(demand=y_test, q=orders_ssnp_ann, alpha=None, u=underage_data_single, o=overage_data_single))
    
    # Integrated Optimization Approach - XGBoost - Simple:
    xgb_model, params, results = tune_XGB_model(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, alpha_input=None, underage_input=underage_data_single, overage_input=overage_data_single, multi=False, integrated=True)
    xgb_result = xgb_model.predict(xgb.DMatrix(X_test))
    profit_simple_XGB_IOA = np.mean(nvps_profit(demand=y_test, q=xgb_result, alpha=None, u=underage_data_single, o=overage_data_single))

    print("XGB - Simple Examination")
    print("Profit Simple XGB IOA: ", profit_simple_XGB_IOA)
    print(y_test.shape, xgb_result.shape)
    print( X_test.shape,y_test.shape, X_train.shape, y_train.shape, X_val.shape, y_val.shape)

    
    print("Step 3")
    
    # Seperated Optimization Approach - XGBoost - Simple:
    xgb_model, hyperparameter_XGB_SOA_Complex, val_profit = tune_XGB_model(X_train, y_train, X_val, y_val, None, underage_data_single, overage_data_single, multi = False, integrated = False)
    target_prediction_XGB = xgb_model.predict(xgb.DMatrix(X_test))
    train_prediction_XGB = xgb_model.predict(xgb.DMatrix(X_train))  
    orders_ssp_XGB, orders_ssnp_XGB = solve_basic_newsvendor_seperate(y_train=y_train, y_train_pred=train_prediction_XGB, y_test_pred=target_prediction_XGB, u=underage_data_single, o=overage_data_single)
    profit_ssp_XGB = np.mean(nvps_profit(y_test, orders_ssp_XGB, alpha_data, underage_data_single, overage_data_single))
    profit_ssnp_XGB = np.mean(nvps_profit(y_test, orders_ssnp_XGB, alpha_data, underage_data_single, overage_data_single))

    # Optimal profit
    optimal_profit_single = np.mean(nvps_profit(demand=y_test, q=y_test, alpha=None, u=underage_data_single, o=overage_data_single))
    

    # Print results
    print("Profit Complex XGB IOA: ", profit_complex_XGB_IOA)
    print("Profit Complex XGB SOA - parametric: ", profit_scp_XGB)
    print("Profit Complex XGB SOA - non-parametric: ", profit_scnp_XGB)
    print("Profit Simple XGB IOA: ", profit_simple_XGB_IOA)
    print("Profit Simple XGB SOA - parametric: ", profit_ssp_XGB)
    print("Profit Simple XGB SOA - non-parametric: ", profit_ssnp_XGB)
    
    # Print results
    print("Profit Single ETS: ", profit_single_ets)
    print("Profit Multi ETS: ", profit_multi_ets)
    print("Profit Complex ANN IOA: ", profit_complex_ANN_IOA)
    print("Profit Complex ANN SOA - parametric: ", profit_scp_ANN)
    print("Profit Complex ANN SOA - non-parametric: ", profit_scnp_ANN)
    print("Profit Simple ANN IOA: ", profit_simple_ANN_IOA)
    print("Profit Simple ANN SOA - parametric: ", profit_ssp_ANN)
    print("Profit Simple ANN SOA - non-parametric: ", profit_ssnp_ANN)

    print("Optimal profit single: ", optimal_profit_single)
    print("Optimal profit multi: ", optimal_profit_multi)

    