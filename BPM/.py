#!/usr/bin/env python
# coding: utf-8

# The goal of this project is to predict the remaining time of an application process. Therefor we use the dataset of the 2018 BPI Challenge which describes applications for EU direct payments for German farmers from the European Agricultural Guarantee Fund. We assume that the last event of an application is the end of the process and therefore is seen as remaining time = 0. 
# 

# In[1]:


from BPM_utils import load_packages, create_environment

load_packages()
create_environment()


# In[2]:


# Standard library imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Related third party imports
import pm4py
import holidays
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import xgboost as xgb
from keras.models import load_model

# Local application/library specific imports
from BPM_utils import tune_NN_model_optuna
from BPM_utils import tune_XGB_model


# # 1. Preparation

# In[3]:


# Load the data from the xes file and convert it to a dataframe (10 minutes)
path = 'BPI Challenge 2018.xes'
data_xes = pm4py.read_xes(path)    
data = pm4py.convert_to_dataframe(data_xes)
data.to_csv('BPI 2018.csv', index=False)


# Example path of an application

# In[21]:


# Examine a specific trace
example_trace = data_xes[0]  # Taking the first trace for demonstration
for i in range(len(example_trace)-1):
    current_event = example_trace[i]
    next_event = example_trace[i+1]
    print(f"Current Event: {current_event['concept:name']}, Next Event: {next_event['concept:name']}")


# In[4]:


data = pd.read_csv('BPI 2018.csv')
# Display column names
print("Column names:")
print(data.columns)


# To execute datadriven regression we need to eliminate all Nan values. These onyl occur for numerical values (we do 0) and for the column note, where we fill in "-" 

# In[5]:


numerical_cols = data.select_dtypes(include=[np.number]).columns
data[numerical_cols] = data[numerical_cols].fillna(0)

data['note'] = data['note'].fillna('-')


# In[6]:


# To check if any NaN values exist in the DataFrame
nan_exists = data.isna().any().any()

print("NaN values exist in the DataFrame:", nan_exists)


# # 1.1 Preparation of timestamp
# 
# The main information we need for this task is the timestamp. Therfore we convert the column with this data to datatime. We will also include following resulting information:
# 
# - date : date of timestamp
# - time : time of timestamp
# - weekday: which weekday is the timestamp from (Monday=0, Tuesday=1, ..., Sunday=6)
# - is_holiday: is this day a federal holiday in germany (boolean)
# 
# To execute the task we will need information about the remaining time of the application process. Therfor we will search for the latest event of each application and mark it with
# 
# - is_latest : boolean which is TRUE if it is the latest event of an application process
# - remaining_time : time to the latest event in the process (in seconds)

# In[7]:


# Identify rows where the timestamp column has the default value
default_timestamp = '1970-01-01T01:00:00+01:00'
mask = data['time:timestamp'] == default_timestamp
print("Number of rows with default timestamp: ", mask.sum())

# Convert the timestamp column to datetime
def convert_timestamp(ts):
    try:
        return pd.to_datetime(ts, format='%Y-%m-%d %H:%M:%S.%f%z')
    except ValueError:
        return pd.to_datetime(ts, format='%Y-%m-%d %H:%M:%S%z')
data['time:timestamp'] = data['time:timestamp'].apply(convert_timestamp)
data['time:timestamp'].describe()


# In[8]:


# Sort the DataFrame by 'time:timestamp' and 'case:application'
data = data.sort_values(['case:application', 'time:timestamp'])

# Create a new column 'is_latest' that is True for the latest event of each application
data['is_latest'] = data.groupby('case:application')['time:timestamp'].transform(lambda x: x == x.max())

# Create a new column 'latest_time' that contains the timestamp of the latest event for each application
data['latest_time'] = data.groupby('case:application')['time:timestamp'].transform(max)

# Create a new column 'remaining_time' that contains the time from each event to the latest event of its application
data['remaining_time'] = data['latest_time'] - data['time:timestamp']

# Convert 'remaining_time' to seconds
data['remaining_time'] = data['remaining_time'].dt.total_seconds()

# Drop the 'latest_time' column
data = data.drop(columns=['latest_time'])


# In[9]:


german_holidays = holidays.Germany()

# Create a new column for the date (without time)
data['date'] = data['time:timestamp'].dt.date

# Create a new column 'time' that contains the time part of each timestamp
data['time'] = data['time:timestamp'].dt.time

# Create a new column 'weekday' that contains the day of the week
data['weekday'] = data['time:timestamp'].dt.day_name()

# Create a new column 'is_holiday' that indicates whether each date is a holiday
data['is_holiday'] = data['date'].apply(lambda x: x in german_holidays)


# For the purpose of this task we will only watch at the events that are complete. For this project, a complete application is one that finishes with a payment. All other applications will be droped. We decided to drop all application that were not completed succefull (abortion or some kind of error) since this should be an exeption and we try to give a remaining time prediction for a normal applicant.
# 
# Now we will drop all applications where the last event is not of activity "finish payment".

# In[10]:


# Get the number of unique applications before the drop
num_apps_before = data['case:application'].nunique()

# Get applications where the last event is not 'finish payment'
apps_not_finish_payment = data[(data['is_latest'] == 1) & (data['activity'] != 'finish payment')]['case:application']

# Drop the events of these applications
data = data[~data['case:application'].isin(apps_not_finish_payment)]

# Get the number of unique applications after the drop
num_apps_after = data['case:application'].nunique()

# Calculate the percentage of applications dropped
percent_dropped = (num_apps_before - num_apps_after) / num_apps_before * 100

print(f'{percent_dropped}% of applications were dropped.')


# In[11]:


# Translate all boolean columns to integers
for col in data.columns:
    if data[col].dtype == 'bool':
        data[col] = data[col].astype(int)


# ## 1.2 Visualization of Data
# 
# First we want to see the events per day over the whold dataset

# In[12]:


# Create a new column 'date' that contains just the date part of each timestamp
data['date'] = data['time:timestamp'].dt.date

# Count the number of events for each day
events_per_day = data.groupby('date').size()

# Plot the number of events per day
events_per_day.plot(kind='line', figsize=(10,4))
plt.title('Number of Events per Day')
plt.xlabel('Date')
plt.ylabel('Number of Events')
plt.show()


# We seem to see that there are some periodic trends within our dataset

# In[13]:


fs = 12
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(18, 4))

# Create new columns for year, month, day, and weekday
data['Year'] = data['time:timestamp'].dt.year
data['Month'] = data['time:timestamp'].dt.month
data['Day'] = data['time:timestamp'].dt.day
data['Weekday'] = data['time:timestamp'].dt.day_name()

# Order the weekdays
cats = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
data['Weekday'] = pd.Categorical(data['Weekday'], categories=cats, ordered=True)

for col, col_name in enumerate(['Year', 'Month', 'Day', 'Weekday']):
    data.groupby(col_name).size().plot(kind='bar', ax=axes[col], legend=False)
    
axes[0].set_ylabel('Count of events', fontsize=fs)

fig.tight_layout()
plt.show()


# In[14]:


# Count the number of events for holidays and non-holidays
events_per_holiday = data.groupby('is_holiday').size()

# Plot the total number of events for holidays and non-holidays
events_per_holiday.plot(kind='bar', figsize=(6,2))
plt.title('Total Number of Events for Holidays and Non-Holidays')
plt.xlabel('Is Holiday')
plt.ylabel('Number of Events')
plt.xticks([0, 1], ['No', 'Yes'])
plt.show()


# As expected, on weekends significantly less events are processed. Also if there is holidays, nearly no events are processed. This confirms the assumption that weekends and holidays are significant for this dataset.
# 
# What is also interesting to see ist that most events seem to happen at the end of each year. This makes sense if we assume that the payment to the farmer is on a yearly basis and the farmers tend to apply at the end of the year. We also have an increase in the middle of the year, it seems like some applications may be on a half-year basis. 
# 
# 

# In[15]:


import numpy as np
import seaborn as sns

# Select only numerical columns
numerical_cols = data.select_dtypes(include=[np.number]).columns

# Calculate correlation
corr = data[numerical_cols].corr()

# Set correlation threshold
threshold = 0.05

# Select only correlations for "remaining_time"
corr_remaining_time = corr['remaining_time']

# Create a mask for values above the threshold
mask = np.abs(corr_remaining_time) > threshold

# Apply the mask to the correlations
significant_corr_remaining_time = corr_remaining_time[mask]

# Remove NaN values
significant_corr_remaining_time.dropna(inplace=True)

# Convert Series to DataFrame for heatmap compatibility
significant_corr_remaining_time = significant_corr_remaining_time.to_frame()

# Plot the heatmap
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(significant_corr_remaining_time, annot=True, annot_kws={'fontsize': 10}, fmt='.2f', ax=ax)


# Here we try to see the significance between remaining_time and other features. Due to the amount of features we filter with a threshold (0.05) and only show the significance between remaining_time and features.
# 
# Remarkable is that there are multiple significant correlations between the remaining time and penalties. Not suprisingly there is also a correlation between actual payment and remaining_time, since when the payment happens this indicates that the process goes to an end. As already seen earlier there is a high correlation between year and remeining time since it seems like many applications are processed at the end of the year. 

# # 2. Predictions
# 
# First of all we split into train, val and test set.

# In[16]:


from sklearn.model_selection import train_test_split

# Get a list of unique applications
applications = data['case:application'].unique()

# Split the applications into train, validation, and test sets
train_apps, test_apps = train_test_split(applications, test_size=0.2, random_state=42)
train_apps, val_apps = train_test_split(train_apps, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

# Use the lists of applications to split the data into train, validation, and test sets
train_data = data[data['case:application'].isin(train_apps)]
val_data = data[data['case:application'].isin(val_apps)]
test_data = data[data['case:application'].isin(test_apps)]


# In[17]:


# Split the train, validation, and test data into X and y
X_train = train_data.drop(['is_latest', 'remaining_time'], axis=1)
y_train = train_data[['is_latest', 'remaining_time']]

X_val = val_data.drop(['is_latest', 'remaining_time'], axis=1)
y_val = val_data[['is_latest', 'remaining_time']]

X_test = test_data.drop(['is_latest', 'remaining_time'], axis=1)
y_test = test_data[['is_latest', 'remaining_time']]


# ### 2.1 Prediction with Keras
# 
# Now we will try to predict the remaining time (and the chance that an event is the latest) of an event with a simple Keras Regressor that we optimize with Optuna.

# In[36]:


# Define preprocessing for categorical features (encode them)
categorical_features = data.select_dtypes(exclude=[np.number]).columns.tolist()
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
      
        ('cat', categorical_transformer, categorical_features)],
    remainder='passthrough')

# Preprocessing on  data
X_train = preprocessor.fit_transform(X_train)
X_val = preprocessor.transform(X_val)
X_test = preprocessor.transform(X_test)


# In[37]:


ann_model, ann_params, status = tune_NN_model_optuna(X_train, y_train, X_val, y_val)
ann_model.save('ann_model.h5') 


# In[ ]:


ann_model = load_model('ann_model.h5')

ann_prediction = ann_model.predict(X_test)


# ### 2.2 Prediction with XGBoost
# 
# 

# In[ ]:


xgb_model, xgb_params, status = tune_XGB_model(X_train, y_train, X_val, y_val)
xgb_model.save_model('xgb_model.json')


# In[ ]:


xgb_model = xgb.Booster()
xgb_model.load_model('xgb_model.json')

xgb_prediction = xgb_model.predict(xgb.DMatrix(X_test))


# # 3. Evaluation

# 
