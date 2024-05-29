from BPM_utils import load_packages, load_pickle_data

load_packages()

# Standard library imports
import pickle
import numpy as np

# Related third party imports
from keras.models import load_model

# Local application/library specific imports
from BPM_utils import create_sequences
from BPM_utils import tune_LSTM_model_optuna


################################################### Functions ####################################################################

if __name__ == "__main__":

    with open( "C:/Users/lanza/Integrated-vs-Seperated-Master-Thesis/BPM/train_val_test.pkl", 'rb') as f:
        train_data, val_data, test_data = pickle.load(f)

    print("Data loaded")

    X_train, X_val, X_test, y_train, y_val, y_test = create_sequences(train_data, val_data, test_data, 3)

    print("Sequences created")

    X_train = np.nan_to_num(X_train)
    y_train = np.nan_to_num(y_train)
    X_val = np.nan_to_num(X_val)
    y_val = np.nan_to_num(y_val)

    

    # Train Keras ANN model
    lstm_model, lstm_params, lstm_value = tune_LSTM_model_optuna(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
    print ("Model trained")
    lstm_model.save('lstm_series_model.h5') 

    print("Model saved")

