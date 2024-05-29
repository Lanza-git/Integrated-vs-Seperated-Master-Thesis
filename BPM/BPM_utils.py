
import os

############################################### Set-up ####################################################################

def load_packages():
    """ Load the required packages for the BPM series project."""
    # General imports
    import subprocess
    import sys

    def install(package):
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

    def uninstall(package):
        subprocess.check_call([sys.executable, "-m", "pip", "uninstall", package, "-y"])

    install('pandas')
    install('scikit-learn')
    install('scikeras<0.13')
    install('numpy')
    install('pulp')
    install('xgboost')
    install('optuna')
    install('optuna-integration')
    install('gurobipy')
    install('matplotlib')
    install('seaborn')
    install('holidays')
    install('pm4py')
    install('tensorflow<2.13')
    install('memory_profiler')
    
    print("Packages loaded successfully")
    

####################################### Data Loading and Preprocessing #########################################

def load_pickle_data(file_path):
    """ Load data from a pickle file.

    Parameters
    ----------
    file_path : str
        Path to the pickle file 

    Returns
    -------
    X_train : np.array
        Training feature data
    y_train : np.array
        Training targets
    X_val : np.array
        Validation feature data
    y_val : np.array
        Validation targets
    X_test : np.array
        Test feature data
    y_test : np.array
        Test targets
    """
    import pickle
    import numpy as np
    with open(file_path, 'rb') as f:
        train_data, val_data, test_data = pickle.load(f)

    # Define the features and target
    X_train = np.array(train_data.drop(columns=['remaining_time']))
    y_train = np.array(train_data['remaining_time'])
    X_val = np.array(val_data.drop(columns=['remaining_time']))
    y_val = np.array(val_data['remaining_time'])
    X_test = np.array(test_data.drop(columns=['remaining_time']))
    y_test = np.array(test_data['remaining_time'])

    return X_train, y_train, X_val, y_val, X_test, y_test

def create_sequences(data_train, data_val, data_test, n_steps=3):
    """"
    Create Sequences for LSTM model from train, val and test data

    Parameters
    ----------
    data_train : pd.DataFrame
        Training data
    data_val : pd.DataFrame
        Validation data
    data_test : pd.DataFrame
        Test data
    n_steps : int
        Number of steps to include in each sequence
        
    Returns
    -------
    sequence_train : np.array
        Training sequences
    sequence_val : np.array
        Validation sequences
    sequence_test : np.array    
        Test sequences

    """
    import numpy as np

    # Create sequences for the training data
    sequence_train = []
    sequence_val = []
    sequence_test = []

    # Create lists to store the remaining_time
    remaining_time_train = []
    remaining_time_val = []
    remaining_time_test = []

    app_ids_train = data_train['case:application'].unique()
    app_ids_train = app_ids_train[:10]
    for app_id in app_ids_train:
        app_data_train = data_train[data_train['case:application'] == app_id]
        app_data_train = app_data_train.sort_values('time:timestamp')
        
        for i in range(len(app_data_train) - n_steps):
            sequence_train.append(app_data_train.iloc[i:i+n_steps].drop(columns=['remaining_time']).values)
            remaining_time_train.append(app_data_train.iloc[i+n_steps-1]['remaining_time'])

    app_ids_val = data_val['case:application'].unique()
    app_ids_val = app_ids_val[:5]
    for app_id in app_ids_val:
        app_data_val = data_val[data_val['case:application'] == app_id]
        app_data_val = app_data_val.sort_values('time:timestamp')
        
        for i in range(len(app_data_val) - n_steps):
            sequence_val.append(app_data_val.iloc[i:i+n_steps].drop(columns=['remaining_time']).values)
            remaining_time_val.append(app_data_val.iloc[i+n_steps-1]['remaining_time'])

    app_ids_test = data_test['case:application'].unique()
    app_ids_test = app_ids_test[:5]
    for app_id in app_ids_test:
        app_data_test = data_test[data_test['case:application'] == app_id]
        app_data_test = app_data_test.sort_values('time:timestamp')
        
        for i in range(len(app_data_test) - n_steps):
            sequence_test.append(app_data_test.iloc[i:i+n_steps].drop(columns=['remaining_time']).values)
            remaining_time_test.append(app_data_test.iloc[i+n_steps-1]['remaining_time'])
    
    sequence_train = np.array(sequence_train)
    sequence_val = np.array(sequence_val)
    sequence_test = np.array(sequence_test)
    remaining_time_test = np.array(remaining_time_test)
    remaining_time_train = np.array(remaining_time_train)
    remaining_time_val = np.array(remaining_time_val)

    return sequence_train, sequence_val, sequence_test, remaining_time_train, remaining_time_val, remaining_time_test

############################################### Neural Network Models ####################################################################

def create_NN_model(n_hidden, n_neurons, activation, input_shape, learning_rate, output_shape, seed=42): 
        """Create a neural network model for time series forecasting

        Parameters
        ----------
        n_hidden : int
            Number of hidden layers
        n_neurons : int
            Number of neurons in each hidden layer
        activation : str
            Activation function for the hidden layers
        input_shape : int
            Number of input units
        learning_rate : float
            Learning rate for the optimizer
        output_shape : int
            Number of output units
        seed : int
            Random seed for reproducibility

        Returns
        -------
        model : tf.keras.Model
            Neural network model
        """
        # TensorFlow and Keras libraries for model creation and training
        from tensorflow.keras.layers import Dense
        from tensorflow.keras.models import Sequential
        import tensorflow as tf
        from tensorflow.keras.layers import Input
        import numpy as np
        
        """ Build a neural network model with the specified architecture and hyperparameters """
        model = Sequential()
        model.add(Input(shape=(input_shape,)))
        for layer in range(n_hidden):
            model.add(Dense(n_neurons, activation=activation))
        model.add(Dense(output_shape))
        # set seed for reproducability
        tf.random.set_seed(seed)
        np.random.seed(seed)
        custom_loss = tf.keras.losses.MeanSquaredError()
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(loss=custom_loss, optimizer=optimizer, metrics=None)
        return model

def create_LSTM_model(n_hidden, n_neurons, activation, input_shape, learning_rate, output_shape, seed=42): 
    """
    Create a Long Short-Term Memory (LSTM) model for time series forecasting

    Parameters
    ----------
    n_hidden : int
        Number of hidden layers
    n_neurons : int
        Number of neurons in each hidden layer
    activation : str
        Activation function for the hidden layers
    input_shape : tuple
        Shape of the input data (timesteps, input_dim)
    learning_rate : float
        Learning rate for the optimizer
    output_shape : int
        Number of output units
    seed : int
        Random seed for reproducibility

    Returns
    -------
    model : tf.keras.Model
        LSTM model

    """
    # TensorFlow and Keras libraries for model creation and training
    from tensorflow.keras.layers import Dense, LSTM
    from tensorflow.keras.models import Sequential
    import tensorflow as tf
    from tensorflow.keras.layers import Input
    import numpy as np
    
    # set seed for reproducability
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
    # Build a neural network model with the specified architecture and hyperparameters
    model = Sequential()
    model.add(Input(shape=input_shape))  # input_shape should be a tuple (timesteps, input_dim)
    model.add(LSTM(n_neurons, activation=activation))
    for layer in range(n_hidden):
        model.add(Dense(n_neurons, activation=activation))
    model.add(Dense(output_shape))
    custom_loss = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss=custom_loss, optimizer=optimizer, metrics=None)
    
    return model

def tune_LSTM_model_optuna(X_train, y_train, X_val, y_val, patience=10, verbose=0, seed=42, threads=40, trials=50):
    """ Tune an LSTM model on the given training data with early stopping using Optuna.

    Parameters
    ----------
    X_train : np.array
        Training feature data (samples, timesteps, features)
    y_train : np.array
        Training targets (samples, 1)
    X_val : np.array
        Validation feature data (samples, timesteps, features)
    y_val : np.array
        Validation targets (samples, 1)
    patience : int
        Number of epochs without improvement before stopping
    verbose : int
        Keras' verbose parameter for silent / verbose model training
    seed : int
        Random seed (affects mainly model initialization, set for reproducable results)
    threads : int
        Number of threads to use for parallel optimization
    trials : int
        Number of trials to run for hyperparameter optimization

    Returns
    -------
    best_model : tf.keras.Model
        Best model found by Optuna
    best_params : dict
        Best hyperparameters found by Optuna
    best_value : float
        Best validation loss found by Optuna

    """

    import optuna
    import numpy as np

    sequence_train = X_train
    remaining_time_train = y_train
    sequence_val = X_val
    remaining_time_val = y_val

    # Determine the input shape and number of output units
    timesteps = sequence_train.shape[1]
    input_dim = sequence_train.shape[2]
    output_shape = 1  # We're predicting a single value

    def objective(trial):
        # Define hyperparameters
        n_hidden = trial.suggest_int('n_hidden', 1, 3)
        n_neurons = trial.suggest_int('n_neurons', 30, 100)
        activation = trial.suggest_categorical('activation', ['relu', 'tanh'])
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
        epochs = trial.suggest_int('epochs', 10, 50)

        # Create the LSTM model
        model = create_LSTM_model(n_hidden, n_neurons, activation, (timesteps, input_dim), learning_rate, output_shape)

        # Compile and train the model
        model.compile(loss='mean_squared_error', optimizer='adam')
        history = model.fit(sequence_train, remaining_time_train, epochs=epochs, validation_data=(sequence_val, remaining_time_val))

        # Evaluate the model
        val_loss = history.history['val_loss'][-1]
        return val_loss

    # Create a study object and optimize the objective
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=trials, n_jobs=threads, gc_after_trial=True)

    # Print the result
    best_model_unfit = create_LSTM_model(study.best_params['n_hidden'], study.best_params['n_neurons'], study.best_params['activation'],
                                    (timesteps, input_dim), study.best_params['learning_rate'], output_shape)
    best_model = best_model_unfit.fit(sequence_train, remaining_time_train, epochs=study.best_params['epochs'] , validation_data=(sequence_val, remaining_time_val))
    best_params = study.best_params
    best_value = study.best_value
    print(f'Best parameters: {best_params}\nBest validation loss: {best_value}')

    return best_model, best_params, best_value


def tune_NN_model_optuna(X_train, y_train, X_val, y_val, lstm = False, patience=10, verbose=0, seed=42, threads=40, trials=100):
    """ Tune a neural network model on the given training data with early stopping using Optuna.
    
    Parameters
    --------------
    X_train : np.array
        training feature data (samples, features)
    y_train : np.array
        training targets (samples, N_PRODUCTS)
    X_val : np.array
        validation feature data (samples, features)
    y_val : np.array
        validation targets (samples, N_PRODUCTS)
    patience : int
        number of epochs without improvement before stopping
    multi : bool
        if True, all products are considered, if False, only product 1 is considered
    integrated : bool
        if True, the optimization is integrated, if False, the optimization is separate
    verbose : int
        keras' verbose parameter for silent / verbose model training
    seed : int
        random seed (affects mainly model initialization, set for reproducable results)

    Returns
    ----------
    best_estimator : keras model
        Best model found by Optuna
    hyperparameter : list
        Best hyperparameters found by Optuna
    study.best_value : float
        Best profit found by Optuna
    """
    import numpy as np
    # TensorFlow and Keras libraries for model creation and training
    from tensorflow.keras.callbacks import EarlyStopping
    # Scikit-learn wrapper for Keras
    from scikeras.wrappers import KerasRegressor
    import optuna
    from optuna.integration import KerasPruningCallback


    output = y_train.shape[1] #N_PRODUCTS
    input = X_train.shape[1] #N_FEATURES
    
    def objective(trial):
        # define the hyperparameters space
        n_hidden = trial.suggest_int('n_hidden', 0, 10)
        n_neurons = trial.suggest_int('n_neurons', 1, 30)
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        epochs = trial.suggest_int('epochs', 10, 50)
        activation = trial.suggest_categorical('activation', ['relu', 'sigmoid', 'tanh'])

        # create a neural network model with basic hyperparameters
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience)

        model_ANN = KerasRegressor(model=create_NN_model, n_hidden=n_hidden, n_neurons=n_neurons, activation=activation,
                                    input_shape=input, learning_rate=learning_rate, output_shape=output, 
                                    seed=seed, callbacks=[early_stopping])

        pruning_callback = KerasPruningCallback(trial, 'val_loss')
        
        model_ANN.fit(X_train, y_train, validation_data=(X_val, y_val),  epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=[pruning_callback]) 

        q_val = model_ANN.predict(X_val)
        result = -np.abs(np.mean(q_val-y_val))
        return result

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=trials, n_jobs=threads, gc_after_trial=True)

    # Get the best parameters and best estimator
    best_params = study.best_params
    best_estimator = train_NN_model([best_params['n_hidden'], best_params['n_neurons'], best_params['learning_rate'],
                                    best_params['epochs'], patience, best_params['batch_size'], best_params['activation']], 
                                    X_train, y_train, X_val, y_val, seed=seed)

    hyperparameter = [best_params['n_hidden'], best_params['n_neurons'],best_params['learning_rate'], 
                    best_params['epochs'], patience, best_params['batch_size'], best_params['activation']]
    
    return best_estimator, hyperparameter, study.best_value


def train_NN_model(hp, X_train, y_train, X_val, y_val, verbose=0, seed=42):
    """ Train a network on the given training data with early stopping.
    
    Parameters
    --------------
    hp : list or tupl
        hyperparameters in the following order: hidden_nodes, lr, max_epochs, patience, batch_size
    X_train : np.array
        training feature data
    y_train : np.array
        training targets
    X_train : np.array
        validation feature data
    y_train : np.array
        validation targets
    alpha : np.array
        Substitution rates, shape (N_PRODUCTS, N_PRODUCTS)
    u : np.array
        underage costs, shape (1, N_PRODUCTS)
    o : np.array
        overage costs, shape (1, N_PRODUCTS)
    verbose : int
        keras' verbose parameter for silent / verbose model training
    seed : int
        random seed (affects mainly model initialization, set for reproducable results)

    Returns
    ----------
    model : keras model
        Final model
    """
    import tensorflow as tf
    import numpy as np
    from tensorflow.keras.callbacks import EarlyStopping

    loss = tf.keras.losses.MeanSquaredError()

    # extract hyperparameters, build and compile MLP
    hidden_nodes, n_neurons, lr, max_epochs, patience, batch_size, activation = hp
    mlp = create_NN_model(n_hidden=hidden_nodes, n_neurons=n_neurons, activation=activation, 
                          input_shape=X_train.shape[1], learning_rate=lr, 
                          output_shape=y_train.shape[1], seed=seed)

    # train MLP with early stopping
    callback = EarlyStopping(monitor='val_loss', patience=patience)
    mlp.fit(X_train, y_train, epochs=max_epochs, batch_size=batch_size, validation_data=(X_val, y_val),
            verbose=verbose, callbacks=[callback])
    
    return mlp

############################################### XGBoost Models ####################################################################

def tune_XGB_model(X_train, y_train, X_val, y_val, patience=10,verbose=0, seed=42, threads=40, trials=100):
    """ Tune an XGBoost model on the given training data with early stopping using Optuna.

    Parameters
    ----------
    X_train : np.array
        Training feature data (samples, features)
    y_train : np.array
        Training targets (samples, 1)
    X_val : np.array
        Validation feature data (samples, features)
    y_val : np.array
        Validation targets (samples, 1)
    patience : int
        Number of epochs without improvement before stopping
    verbose : int
        Keras' verbose parameter for silent / verbose model training
    seed : int
        Random seed (affects mainly model initialization, set for reproducable results)
    threads : int
        Number of threads to use for parallel optimization
    trials : int
        Number of trials to run for hyperparameter optimization
        
    Returns
    -------
    best_model : xgb.Booster
        Best model found by Optuna
    best_params : dict
        Best hyperparameters found by Optuna
    best_value : float
        Best validation loss found by Optuna

    """
    import xgboost as xgb
    import optuna
    import numpy as np

    if y_train.shape[1] == 1:
        multi_strategy = "one_output_per_tree"
        custom_objective = 'reg:squarederror'
    elif y_train.shape[1] != 1:
        multi_strategy = "multi_output_tree"
        custom_objective = "reg:squarederror"
    else:
        raise ValueError('Invalid Configuration')

    X, y = X_train, y_train  
    Xy = xgb.DMatrix(X, label=y)
    dval = xgb.DMatrix(X_val, label=y_val)
    results = {}
    def objective(trial):
        params = {
                "tree_method": "hist",
                "num_target": y.shape[1],
                "multi_strategy": multi_strategy,
                "learning_rate": trial.suggest_float("learning_rate", 0.1, 0.5),
                "max_depth": trial.suggest_int("max_depth", 2, 6),
                "n_estimators": trial.suggest_int("n_estimators", 100, 140),    #????????
                "subsample": trial.suggest_float("subsample", 0.3, 0.9),
                "objective": custom_objective,
        }
        booster = xgb.train(
            params,
            dtrain=Xy,
            num_boost_round=128,
            evals=[(dval, "val")],
            evals_result=results,
            early_stopping_rounds=patience,
            )
       
         # make predictions on validation set and compute profits
        val_set = xgb.DMatrix(X_val)
        q_val = booster.predict(val_set)

        result = -np.abs(np.mean(q_val-y_val))

        return result

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trials, n_jobs=threads)  

    trial = study.best_trial
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Get the best parameters
    best_params = study.best_trial.params

    # Add the fixed parameters
    best_params.update({
        "tree_method": "hist",
        "num_target": y.shape[1],
        "multi_strategy": multi_strategy,
        "objective": custom_objective,
    })

    # Train the final model
    final_booster = xgb.train(
        best_params,
        dtrain=Xy,
        num_boost_round=128,
        evals=[(dval, "val")],
    )

    return final_booster, best_params, results