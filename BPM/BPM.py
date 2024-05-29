from BPM_utils import load_packages, load_pickle_data

load_packages()

# Standard library imports
import pickle
import numpy as np

# Related third party imports
from keras.models import load_model

# Local application/library specific imports
from BPM_utils import tune_NN_model_optuna
from BPM_utils import tune_XGB_model

#################################################### Functions ####################################################################

if __name__ == "__main__":

    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_pickle_data('/pfs/data5/home/ma/ma_ma/ma_elanza/bpm_dir/train_val_test.pkl')
    del X_test, y_test

    X_train_input = np.nan_to_num(X_train)
    y_train_input = np.nan_to_num(y_train)
    X_val_input = np.nan_to_num(X_val)
    y_val_input = np.nan_to_num(y_val)

    # Train Keras ANN model
    ann_model, ann_params, status = tune_NN_model_optuna(X_train=X_train_input, y_train=y_train_input, X_val=X_val_input, y_val=y_val_input)
    ann_model.save('ann_model.h5') 

    # Train XGBoost model
    xgb_model, xgb_params, status = tune_XGB_model(X_train_input, y_train_input, X_val_input, y_val_input)
    xgb_model.save_model('xgb_model.json')

