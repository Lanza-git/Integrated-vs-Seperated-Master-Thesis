from BPM_utils import load_packages, load_pickle_data

load_packages(optuna=False)

# Standard library imports
import pickle
import numpy as np

# Related third party imports
from keras.models import load_model

# Local application/library specific imports
from BPM_utils import create_sequences, sequence_to_lagged_features
from BPM_utils import tune_LSTM_model_optuna, tune_XGB_model


################################################### Functions ####################################################################

if __name__ == "__main__":
    
    # Define sequence length
    sequence_length = 3
    limit = 1000

    with open( "/pfs/data5/home/ma/ma_ma/ma_elanza/bpm_dir/train_val_test.pkl", 'rb') as f:
        train_data, val_data, test_data = pickle.load(f)

    print("Data loaded")

    X_train, X_val, X_test, y_train, y_val, y_test = create_sequences(data_train=train_data, data_val=val_data, data_test=test_data, n_steps=sequence_length, limit=limit)

    # Save the sequence data
    with open('sequence_data.pkl', 'wb') as f:
        pickle.dump((X_train, X_val, X_test, y_train, y_val, y_test), f)

    print("Sequences created")

    X_train = np.nan_to_num(X_train)
    y_train = np.nan_to_num(y_train)
    X_val = np.nan_to_num(X_val)
    y_val = np.nan_to_num(y_val)

    
    # Train Keras ANN model
    lstm_model, lstm_params, lstm_value = tune_LSTM_model_optuna(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, trials=10)
    print ("Model trained")
    lstm_model.save('/pfs/data5/home/ma/ma_ma/ma_elanza/bpm_dir/lstm_series_model.h5') 

    print("Model saved")

    
 
    # Convert the sequence data to 2D data with laggend Feature
    lagged_train_2D = sequence_to_lagged_features(X_train, (sequence_length-1))
    lagged_val_2D = sequence_to_lagged_features(X_val, (sequence_length-1))
    lagged_test_2D = sequence_to_lagged_features(X_test, (sequence_length-1))
    y_train = y_train.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)    
    y_test = y_test.reshape(-1, 1)

    # Save all the data into one file
    with open('lagged_data.pkl', 'wb') as f:
        pickle.dump((lagged_train_2D, lagged_val_2D, lagged_test_2D, y_train, y_val, y_test), f)

    # Train XGBoost model
    xgb_model, xgb_params, status = tune_XGB_model(X_train=lagged_train_2D, y_train=y_train, X_val=lagged_val_2D, y_val=y_val, trials=10)
    print("XGB Model trained")

    # Save the model
    xgb_model.save_model('/pfs/data5/home/ma/ma_ma/ma_elanza/bpm_dir/xgb_series_model.json')
    print("Model saved")
