# General Imports
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

######################## Data Handling Functions ############################################################

def load_data(path, multi=False, test_size=0.2):

    """ Load  data for the newsvendor problem from specified location 

    Parameters
    ---------
    path : str
    multi : bool        - if True, all products are considered, if False, only product 1 is considered
    test_size : float   - proportion of the dataset to include in the test split

    Returns
    ---------
    X_train, X_test, target_train, target_test: np.arrays

    """ 
    # Load Data
    raw_data = pd.read_csv(path)    

    # Select only one product if multi == False
    if multi == False:
        # Select only columns with product_1_demand or not demand (features)
        selected_columns = raw_data.columns[raw_data.columns.str.contains('product_1_demand') | ~raw_data.columns.str.contains('demand')]
        raw_data = raw_data[selected_columns]

    # Split the data into feature and target data
    feature_columns = raw_data.columns[raw_data.columns.str.contains('demand') == False]
    feature_data = raw_data[feature_columns]
    target_columns = raw_data.columns[raw_data.columns.str.contains('demand')]
    target_data = raw_data[target_columns]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(feature_data, target_data, test_size=test_size, random_state=42)

    return X_train, X_test, y_train, y_test


def preprocess_data(X_train, X_test, target_train, target_test):

    """ Preprocess the data for the newsvendor problem
    
    Parameters
    ---------
    X_train, X_test, target_train, target_test: np.arrays

    Returns
    ---------
    X_train_p, X_test_p, target_train_p, target_test_p: np.arrays    
    
    """

    # Define preprocessing for numeric columns (scale them)
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    # Define preprocessing for categorical features (encode them)
    categorical_features = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
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
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    return X_train, X_test, target_train, target_test