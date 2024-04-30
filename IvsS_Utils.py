# Data manipulation libraries
import numpy as np
import pandas as pd

# Scikit-learn libraries for data preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Scikit-learn libraries for model selection and evaluation
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, train_test_split

# Scikit-learn library for pipeline creation
from sklearn.pipeline import Pipeline

# TensorFlow and Keras libraries for model creation and training
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# Scikit-learn wrapper for Keras
from scikeras.wrappers import KerasRegressor

# SciPy library for statistical functions
from scipy.stats import reciprocal

# pulp for mathematical optimization
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpBinary, LpStatus


######################## Data Handling Functions ############################################################

def load_data(path, multi=False):

    """ Load  data for the newsvendor problem from specified location 

    Parameters
    ---------
    path : str
    multi : bool        - if True, all products are considered, if False, only product 1 is considered
    test_size : float   - proportion of the dataset to include in the test split

    Returns
    ---------
    raw_data : np.array

    """ 
    # Load Data
    raw_data = pd.read_csv(path)    

    # Select only one product if multi == False
    if multi == False:
        # Select only columns with product_1_demand or not demand (features)
        selected_columns = raw_data.columns[raw_data.columns.str.contains('product_1_demand') | ~raw_data.columns.str.contains('demand')]
        raw_data = raw_data[selected_columns]

    raw_data = raw_data.to_numpy()
    return raw_data

def preprocess_data(raw_data):

    """ Preprocess the data for the newsvendor problem
    
    Parameters
    ---------
    raw_data : np.array

    Returns
    ---------
    feature_data, target_data: np.arrays    
    
    """

        # Split the data into feature and target data
    feature_columns = raw_data.columns[raw_data.columns.str.contains('demand') == False]
    feature_data = raw_data[feature_columns]
    target_columns = raw_data.columns[raw_data.columns.str.contains('demand')]
    target_data = raw_data[target_columns]

    # Define preprocessing for numeric columns (scale them)
    numeric_features = feature_data.select_dtypes(include=[np.number]).columns.tolist()
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    # Define preprocessing for categorical features (encode them)
    categorical_features = feature_data.select_dtypes(exclude=[np.number]).columns.tolist()
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)],
        remainder='passthrough')

    # Preprocessing on  data
    feature_data = preprocessor.fit_transform(feature_data)
    
    return feature_data, target_data

def split_data(feature_data, target_data, test_size=0.2, val_size=0.2):
    # First, split the data into training+validation set and test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(feature_data, target_data, test_size=test_size, random_state=42)

    # Then, split the training+validation set into training set and validation set
    val_size_adjusted = val_size / (1 - test_size)  # Adjust the validation size
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size_adjusted, random_state=42)

    # Convert to numpy arrays if they are pandas DataFrames
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.values
    if isinstance(X_val, pd.DataFrame):
        X_val = X_val.values
    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.values
    if isinstance(y_train, pd.DataFrame) or isinstance(y_train, pd.Series):
        y_train = y_train.values
    if isinstance(y_val, pd.DataFrame) or isinstance(y_val, pd.Series):
        y_val = y_val.values
    if isinstance(y_test, pd.DataFrame) or isinstance(y_test, pd.Series):
        y_test = y_test.values

    return X_train, y_train, X_val, y_val, X_test, y_test



######################### Newsvendor related Functions ########################################################

def nvps_profit(demand, q, alpha, u, o):

    """ Profit function of the newsvendor under substitution
    
    Parameters
    ---------
    demand : np.array
        demand, shape (T, N_PRODUCTS)
    q : np.array
        orders, shape (T, N_PRODUCTS)
    alpha: np.array
        substitution rates, shape (N_PRODUCTS, N_PRODUCTS)
    u : np.array
        underage costs, shape (1, N_PRODUCTS)
    o : np.array
        overage costs, shape (1, N_PRODUCTS)

    Returns
    ---------
    profits: np.array
        Profits by period, shape (T,1)
    """
    if demand.shape[1] == 1:
        q = np.maximum(0., q)
        profits = np.maximum(q-demand, 0.)*u - np.maximum(demand-q, 0.)*(u+o)
    else:
        q = np.maximum(0., q) # make sure orders are non-negative
        demand_s = demand + np.matmul(np.maximum(demand-q, 0.), alpha) # demand including substitutions
        profits = np.matmul(q, u.T) - np.matmul(np.maximum(q-demand_s, 0.), (u+o).T) # period-wise profit (T x 1)
    return profits

def solve_MILP(d, alpha, u, o):

    """ helper function that solves the mixed-integer linear program (MILP) of the multi-product newsvendor problem under substitution (cf. slides)
    
    Parameters
    -----------
    d : np.array
        Demand samples of shape (n, N_PRODUCTS), where n denotes the number of samples
    alpha : np.array
        Substitution rates, shape (N_PRODUCTS, N_PRODUCTS)
    u : np.array
        underage costs, shape (1, N_PRODUCTS)
    o : np.array
        overage costs, shape (1, N_PRODUCTS)
    n_prods : int
        number of products

    Returns
    ----------
    orders: np.array
        Optimal orders, of shape (1, N_PRODUCTS)
    model.status : int
       
    """

    n_prods = d.shape[1] # number of products

    hist = d.shape[0] # number of demand samples 

    # compute upper bounds M
    d_min = np.min(d, axis=0)
    d_max = d + np.matmul(d, alpha)
    M = np.array(np.max(d_max, axis=0)[0]-d_min)

    # initialize model
    model = LpProblem("My Model", LpMaximize)

    # initialize model variables
    z = [[LpVariable(f'z_{t}_{i}', cat='Binary') for i in range(n_prods)] for t in range(hist)]
    q = [LpVariable(f'q_{i}') for i in range(n_prods)]
    y = [[LpVariable(f'y_{t}_{i}') for i in range(n_prods)] for t in range(hist)]
    v = [[LpVariable(f'v_{t}_{i}') for i in range(n_prods)] for t in range(hist)]

    # objective function
    model += lpSum(u[0, i].item()*q[i] - (u[0, i].item()+o[0, i].item()) / hist * y[t][i] for i in range(n_prods) for t in range(hist))

    # constraints
    for i in range(n_prods):
        for t in range(hist):
            model += y[t][i] >= q[i] - d[t, i].item() - lpSum(alpha[j, i]*v[t][j] for j in range(n_prods))
            model += v[t][i] <= d[t, i].item() - q[i] + M[i]*z[t][i]
            model += v[t][i] >= d[t, i].item() - q[i] - M[i]*z[t][i]
            model += v[t][i] <= d[t, i].item() * (1 - z[t][i])

    # solve and retrieve solution
    model.solve()
    orders = np.array([[q[p].varValue for p in range(n_prods)]])

    return orders, LpStatus[model.status]

######################### Neural Network Functions ######################################################################

def make_nvps_loss(alpha, underage, overage):

    """ Create a custom loss function for the newsvendor problem under substitution"""

    # transofrm the alpha, u, o to tensors
    underage = tf.convert_to_tensor(underage, dtype=tf.float32) #underage costs
    overage = tf.convert_to_tensor(overage, dtype=tf.float32) #overage costs
    alpha = tf.convert_to_tensor(alpha, dtype=tf.float32) #substitution matrix

    # define the loss function
    @tf.autograph.experimental.do_not_convert
    def nvps_loss(y_true, y_pred):
        q = tf.maximum(y_pred, 0.)

        # Calculate the demand increase for each product due to substitutions from other products
        demand_increase = tf.matmul( tf.maximum(0.0, y_true - y_pred),alpha)
        # Adjusted demand is the original demand plus the increase due to substitutions
        adjusted_demand = y_true + demand_increase
        # Calculate the profits
        profits = tf.matmul(q,tf.transpose(underage)) - tf.matmul(tf.maximum(0.0,q - adjusted_demand), tf.transpose(underage+overage))

        return -tf.math.reduce_mean(profits)
    
    return nvps_loss

def make_nvp_loss(underage, overage):

    """ Create a custom loss function for the newsvendor problem without substitution"""

    q = underage / (underage + overage)

    @tf.autograph.experimental.do_not_convert
    def nvp_loss(y_true, y_pred):
        error = y_true - y_pred
        return tf.keras.backend.mean(tf.maximum(q*error, (q-1)*error), axis=-1)
    
    return nvp_loss

def create_NN_model(n_hidden, n_neurons, activation, input_shape, learning_rate, loss, output_shape, seed=42): 
        
        """ Build a neural network model with the specified architecture and hyperparameters """

        model = Sequential()
        model.add(Dense(n_neurons, activation=activation, input_shape=input_shape))
        for layer in range(n_hidden):
            model.add(Dense(n_neurons, activation=activation))
        model.add(Dense(output_shape))

            # set seed for reproducability
        tf.random.set_seed(seed)
        np.random.seed(seed)
    
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(loss=loss, optimizer=optimizer)

        return model

def tune_NN_model(X_train, y_train, X_val, y_val, alpha, underage, overage, patience=10, integrated=True, verbose=0, seed=42):

    """ Train a network on the given training data with hyperparameter tuning
    
    Parameters
    --------------
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
    patience : int
        number of epochs without improvement before stopping training
    verbose : int
        keras' verbose parameter for silent / verbose model training
    seed : int
        random seed (affects mainly model initialization, set for reproducable results)

    Returns
    ----------
    model : keras model
        Final model
    hp : list or tupl
        hyperparameters in the following order: hidden_nodes, lr, max_epochs, patience, batch_size
    val_profit : float
        Mean profit on the validation set
    """

    columns = y_train
    # construct loss function based on the number of products
    if integrated == False:
        loss = tf.keras.losses.MeanSquaredError()
    elif (y_train.shape[1] == 1) & (integrated == True):
        loss = make_nvp_loss(underage, overage)
    elif (y_train.shape[1] > 1) & (integrated == True): 
        loss = make_nvps_loss(alpha, underage, overage)
    else:
        raise ValueError('Invalid loss function')

    # create a neural network model with basic hyperparameters
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience)
    model_ANN = KerasRegressor(build_fn=create_NN_model, n_hidden=1,n_neurons=30, activation = 'relu',
                               input_shape=X_train.shape[1], learning_rate=0.01, 
                               loss=loss, output_shape=y_train.shape[1], seed = seed, verbose=verbose, 
                               callbacks=[early_stopping])
    
    # define the hyperparameters space
    param_distribs = {
        "n_hidden": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "n_neurons": np.arange(1, 100),
        "learning_rate": np.logspace(-4,-2,num=10),
        "batch_size": [16, 32, 64, 128],
        "epochs": [10, 20, 30, 40, 50],
        "activation": ['relu', 'sigmoid', 'tanh']
    }

    # perform GridSearch for hyperparameter tuning
    grid = RandomizedSearchCV(estimator=model_ANN, param_distributions=param_distribs, cv=3, n_iter=100, n_jobs=-1, verbose=verbose)
    grid_result = grid.fit(X_train, y_train, validation_data=(X_val, y_val), verbose=verbose)

    # Get the best parameters and best estimator
    best_params = grid_result.best_params_
    best_estimator = grid_result.best_estimator_

    # make predictions on validation set and compute profits
    q_val = best_estimator.predict(X_val).numpy()
    val_profit = np.mean(nvps_profit(y_val, q_val, alpha, underage, overage))

    hyperparameter = [best_params['n_hidden'], best_params['n_neurons'],best_params['learning_rate'], 
                      best_params['epochs'], patience, best_params['batch_size'], best_params['activation']]

    return best_estimator, hyperparameter, val_profit

def train_NN_model(hp, X_train, y_train, X_val, y_val, alpha, underage, overage, integrated=True, verbose=0, seed=42):

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

    # construct loss function based on the number of products
    if integrated == False:
        loss = tf.keras.losses.MeanSquaredError()
    elif (y_train.shape[1] == 1) & (integrated == True):
        loss = make_nvp_loss(underage, overage)
    elif (y_train.shape[1] > 1) & (integrated == True): 
        loss = make_nvps_loss(alpha, underage, overage)

    # extract hyperparameters, build and compile MLP
    hidden_nodes, n_neurons, lr, max_epochs, patience, batch_size, activation = hp
    mlp = create_NN_model(n_hidden=hidden_nodes, n_neurons=n_neurons, activation=activation, 
                          input_shape=X_train.shape[1], learning_rate=lr, loss=loss, 
                          output_shape=y_train.shape[1], seed=seed)

    # train MLP with early stopping
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    mlp.fit(X_train, y_train, epochs=max_epochs, batch_size=batch_size, validation_data=(X_val, y_val),
            verbose=verbose, callbacks=[callback])
    
    return mlp


######################### LightGBM Functions ######################################################################

