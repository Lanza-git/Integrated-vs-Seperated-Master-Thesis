# Comparative Analysis of Separated and Integrated Approaches in Data-Driven Supply Chain Optimization
### Master Thesis of Enrico Lanza at the Chair of Artificial Intelligence (University of Mannheim)

This thesis provides a thorough comparative analysis of Separated and Integrated Optimization Approaches (SOA and IOA) in data-driven supply chain optimization, with a specific focus on inventory management. The study addresses the challenge of managing demand uncertainty in supply chains, where traditional model-driven methods often struggle due to the absence of accurate probability distributions. By using large datasets, data-driven approaches offer a promising alternative.

The research assesses the performance of SOA and IOA across various scenarios. The Separated Approach estimates demand based on historical data and then solves the optimization problem. In contrast, the Integrated Approach combines these steps to solve the optimization problem directly using historical data. 

The experimental evaluation utilizes datasets with varying characteristics and two distinct optimization problems. The research employs Artificial Neural Networks and Extreme Gradient Boosting Decision Trees for demand estimation in SOA and uses them with custom loss functions for optimization in IOA. Results indicate that while IOA is generally more stable to complex data distributions or sparse data, SOA remains competitive in scenarios with higher complexity regarding the inventory problem that has to be solved. The findings provide significant insights into the applicability of each approach, guiding practitioners in selecting the appropriate method based on specific supply chain conditions.

## 01 Data Generation
### Install
The file "IvsS_Data_Generator.py" will automatically install the following packages via "pip", when executed with the latest python version:
- pandas
- numpy
- scikit-learn
- h5py

### Files
- IvsS_Data_Generator.py: File to generate artifical datasets

### Run
The file can be executed on your local device using a python environment.
To characteristics of the generated dataset can be defined within **IvsS_Data_Generator.py**, please refer to the file for additional information.

The output of this code will be a dataset saved as a .h5 file, that is split up into: *X_train, y_train, X_val, y_val, X_test, y_test*.
The saving location of the file can be defined in **IvsS_Data_Generator.py**.
The name of the file will look like *set_430650_data.h5*, where *set_430650* is an example for a dataset ID. The file will be saved in a folder named equal to the dataset ID.
For the creation rules of the dataset ID please refer to the code.

## 02 Main
### Install
When executing the file "IvsS_Main.py", it will automatically install the following packages:
- pandas
- scikit-learn
- scikeras
- numpy
- pulp
- xgboost
- typing
- optuna
- optuna-integration
- gurobipy
- statsmodel
- tensorflow
- psutils

### Structure
- IvsS_Main.py:                   Main File to train all models on a given dataset
- IvsS_Utils.py:                  Utility File containing all functions necessary for IvsS_Main.py
- run_main.sh:                    SH-File used to run IvsS_Main.py file on the BWhpc-Server
  
### Run
The file is optimized for the use on the BWhpc-Server. Nethertheless, it is possible to run the file on your local machine, which is not recommended for larger Datasets due to the excessive ressources needed.

To execute the separated part of the SOA, regarding the complex problem, a gurobi license is needed. 
The location of the license file has to be defined in the "create_environment" function of the **IvsS_Utils.py** file.
Moreover, the path to the repository, containing the dataset, must be defined in **IvsS_Main.py**.
The code expects each dataset to be stored in a folder that is named equal to the dataset ID.

To run **IvsS_Main.py** on your local system, two System Arguments have to be given. First the dataset ID (example: set_430650), than the Risk Factor (example: 1.0):
```
python IvsS_Main.py "set_430650" "1.0"
```
If you utilize a hpc-Server, you can use the provided **run_main.sh** file, where all necessary inputs can be defined.

The the trained models and their metadata file will be stored in the folder containing the dataset.

## 03 Evaluation & Visualization
### Install
To use the files in Evaluaton & Visualization, the following packages are necessary:
- pandas
- numpy
- h5py
- scipy
  
### Structure
- IvsS_Evaluation.ipynb:          Evaluation notebook used to visualize the results of the trained models
- IvsS_Data_Visualization.ipynb:  Notebook to visualize the basic dataset
- IvsS_Evaluation_Utils.py:       Utility File for the evaluation and visualization part

### Run
The notebooks can be executed on your local device and will visualize the trained models and the datasets stored in folders **04 Data** and **05 Models**.

## Contact
For any inquiries or issues, please contact Enrico Lanza at enrico.lanza@students.uni-mannheim.de
