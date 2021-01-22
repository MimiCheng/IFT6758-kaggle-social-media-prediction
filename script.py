#!/usr/bin/env python
# coding: utf-8

# 

# In[1]:


# Team Name- Student Baseline

# Team Members
# Balaji Balasubramanian
# Patcharin Cheng 
# Arjun Vaithilingam Sudhakar

# This is the code for the model that can reproduce our best score.

# Import libraries
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import os
import random
import warnings
from mlxtend.regressor import StackingCVRegressor
from xgboost import XGBRegressor
warnings.filterwarnings('ignore')

# Set random seeds (for reproducibility requirement)
os.environ['PYTHONHASHSEED']=str(1)
np.random.seed(1)
random.seed(1)
    

# Converting the sparse user time zone column into 7 unique categories
time_zone_dict = {
'Eastern Time (US & Canada)':'USA',
'Pacific Time (US & Canada)':'USA',
'Central Time (US & Canada)':'USA',
'Central Time (US & Canada)':'USA',
'London':'Europe',
'Brasilia':'Latin America',
'Paris':'Europe',
'Quito':'Latin America',
'Jakarta':'Asia',
'Amsterdam':'Europe',
'Mexico City':'Europe',
'Madrid':'Europe',
'New Delhi':'Asia',
'Istanbul':'Middle East',
'Hawaii':'USA',
'Tokyo':'Asia',
'Rome':'Europe',
'Santiago':'Latin America',
'Greenland':'Europe',
'Buenos Aires':'Europe',
'Mountain Time (US & Canada)':'USA',
'Riyadh':'Middle East',
'Caracas':'Latin America',
'Athens':'Europe',
'Atlantic Time (Canada)':'USA',
'Bern':'Europe',
'Alaska':'USA',
'Arizona':'USA',
'Bogota':'Latin America',
'Mumbai':'Asia',
'India':'Asia',
'Berlin':'Europe',
'Hong Kong':'Asia',
'Seoul':'Asia',
'Pretoria':'Africa',
'Sydney':'Asia',
'Muscat':'Middle East',
'Baghdad':'Middle East',
'Dublin':'Europe',
'Berlin':'Europe',
'Casablanca':'Africa',
'Cairo':'Africa',
'Abu Dhabi':'Middle East',
'Chennai':'Asia',
'Kuwait':'Middle East',
'Kuala Lumpur':'Asia',
'Brussels':'Europe',
'Moscow':'Asia',
'Central America':'Latin America',
'Ljubljana':'Europe',
'Singapore':'Asia',
'Melbourne':'Asia'}


#Removing 8 columns that are not being used
def drop_columns(df):
    df.drop(['Id','User Name','Location','UTC Offset','Profile Image','Profile Text Color',
               'Profile Page Color','Profile Theme Color'],axis=1,inplace=True)
    
def location_fix(df):
    '''
    this function is to replace city with continent
    '''
    for i in time_zone_dict.items():
        df['User Time Zone'] = df['User Time Zone'].replace(i[0], i[1])

    top_used_loc=['USA','Europe','Latin America','Asia','Middle East','Africa']
    df['User Time Zone'][~df['User Time Zone'].isin(top_used_loc)]='Others'
    

def preprocessing_num(df):
    # Converting personal url to binary
    df['Personal URL'].fillna(0,inplace=True)
    df['Personal URL'][df['Personal URL']!=0]=1

    # Converting '??' from the Location Public Visibility to enabled
    df['Location Public Visibility']=df['Location Public Visibility'].str.lower()
    df['Location Public Visibility']=df['Location Public Visibility'].replace('??','enabled')
    
    # These four languages are the most common. Other languages are converted to 'others'
    top_used_lang=['en','es','pt','fr']
    df['User Language'][~df['User Language'].isin(top_used_lang)]='others'

    # ' ' value in Profile Category  column is converted to 'unkown'
    df['Profile Category']=df['Profile Category'].replace(' ','unknown')
    
    # Here we do a log transform for four continuous valued inputs to remove the skew in the features and 
    # get feature values that resembles a normal distribution.

    df['Num of Followers']= np.log10(1+df['Num of Followers'])
    df['Num of People Following']= np.log10(1+df['Num of People Following'])
    df['Num of Status Updates']= np.log10(1+df['Num of Status Updates'])
    df['Num of Direct Messages']= np.log10(1+df['Num of Direct Messages'])

    
    # We do a log transform of the 'Avg Daily Profile Visit Duration in seconds' column and also impute the 
    # NaN values by the mean value of the column.
    df['Avg Daily Profile Visit Duration in seconds']=np.log10(1+df['Avg Daily Profile Visit Duration in seconds'])
    df['Avg Daily Profile Visit Duration in seconds'].fillna((df['Avg Daily Profile Visit Duration in seconds'].mean()), inplace=True)

    # Same procedure is done for 'Avg Daily Profile Clicks' column also
    df['Avg Daily Profile Clicks']= np.log10(1+df['Avg Daily Profile Clicks'])
    df['Avg Daily Profile Clicks'].fillna((df['Avg Daily Profile Clicks'].mean()), inplace=True)

    # We fill the NaN values in 'Profile Cover Image Status' column by 'Not set'
    df['Profile Cover Image Status'].fillna('Not set',inplace=True)
    


def preprocessing_category(df):


    # Now we convert the categorical column values from text form to numerical form to input it to the model
    cleanup_nums = {"Personal URL": {"0":0, "1":1},
                "Profile Cover Image Status":     {"Not set": 0, "Set": 1},
                "Profile Verification Status": {"Not verified": 0, "Pending": 1, "Verified": 2 },
                "Is Profile View Size Customized?":{"False":0,"True":1},
                "Location Public Visibility":{'disabled':0,'enabled':1},
                "Profile Category":{'unknown':0,'government':1,"business":2,'celebrity':3},
                "User Time Zone":{'Others':0,'Africa':1,'Middle East':2,'Asia':3,'Latin America':4,'Europe':5,'USA':6},
                'User Language':{'others':0,'fr':2,'pt':3,'es':4,'en':5}
               }

    # Converting the data type of the categorical columns to 'str'
    df['Profile Cover Image Status'] = df['Profile Cover Image Status'].astype(str)
    df['Profile Verification Status'] = df['Profile Verification Status'].astype(str)
    df['Is Profile View Size Customized?'] =df['Is Profile View Size Customized?'].astype(str)
    df['Location Public Visibility'] = df['Location Public Visibility'].astype(str)

    df = df.replace(cleanup_nums)
    return df



# Creating 8 new columns from the existing features.
def new_columns(df):
    # Convert the time stamp column into a new column that represents the number of months 
    # the person has been on social media
    df['Profile Creation Timestamp'] = df['Profile Creation Timestamp'].astype(str)
    df['Profile Creation Timestamp'] =pd.to_datetime(df['Profile Creation Timestamp'])
    df['MonthsInSocialMedia'] = ((2020- df['Profile Creation Timestamp'].dt.year) * 12 +
    (11 - df['Profile Creation Timestamp'].dt.month))
    
    ### new columns
    df['MonthsInSocialMedia'] =np.log10(1+df['MonthsInSocialMedia'])    
        
    df['Months follower ratio']=df['Num of Followers']/df['MonthsInSocialMedia']
    df['Months following ratio']=df['Num of People Following']/df['MonthsInSocialMedia']
    df['Months status ratio']=df['Num of Status Updates']/df['MonthsInSocialMedia']
    df['Months messages ratio']=df['Num of Direct Messages']/df['MonthsInSocialMedia']
    group_col = df[['Num of Followers', 'Num of People Following', 'Num of Status Updates', 
                    'Num of Direct Messages',
                    'Avg Daily Profile Visit Duration in seconds', 'Avg Daily Profile Clicks']]
    df['group_sum'] = np.sum(group_col, axis=1)
    df['group_sum']=df['group_sum']/6
    
    df['Total Activity']=df['Num of Status Updates']+df['Num of Direct Messages']
    df['Total clicks from inception']=df['Avg Daily Profile Clicks']*30*train_loc['MonthsInSocialMedia']


# Load data
# Use this if you are running it on kaggle notebook.
train = pd.read_csv('../input/ift6758-a20/train.csv')
test = pd.read_csv('../input/ift6758-a20/test.csv')

tid=test['Id']
test_id=tid.to_numpy()

train_x=train.iloc[:,:24]
train_y=train.iloc[:,23]

train_loc=train_x.copy()

# We dropped irrelavant and sparse columns
drop_columns(train_loc)
drop_columns(test)

# We replaced city with continent
location_fix(train_loc)
location_fix(test)

#data cleaning and preprocessing steps
preprocessing_num(train_loc)
preprocessing_num(test)

train_loc=preprocessing_category(train_loc)
test=preprocessing_category(test)
# We add new columns
new_columns(train_loc)
new_columns(test)

# We drop the 'Profile Creation Timestamp' because we extracted the useful information 
# from this column and stored it in 'MonthsInSocialMedia' column.

train_loc.drop('Profile Creation Timestamp',axis=1,inplace=True)
test.drop('Profile Creation Timestamp',axis=1,inplace=True)

# Drop Label from the training features.
train_loc.drop(['Num of Profile Likes'],axis=1,inplace=True)

# Training features has been stored in a different variable for convenience.
fit_x_all =train_loc.copy()

# Performing log10 transform on the labels to bring it on the same scale as input features.
fit_y_all = np.log10(1+train_y)

# Building a SVR model with RBF Kernel
svr = SVR(kernel='rbf', epsilon=0.2,C=0.75)

# Building XGBRegressor
xgboost = XGBRegressor(learning_rate=0.03,
                       n_estimators=250,
                       max_depth=3,
                       seed=27,
                       alpha=2,
                       random_state=1)

# Building Stacking regressor model with base model as xgboost and svr and the meta regressoras xgboost
stack = StackingCVRegressor(regressors=(xgboost, svr),
                                meta_regressor=xgboost,
                                use_features_in_secondary=True,random_state=15)

# Perform Standard Scaler on the input columns to shift the data distribution to have common scale.
# Creating a pipeline with standard scaler and the model and then fitting it on the training data.
stack_te = make_pipeline(StandardScaler(), stack).fit(fit_x_all, fit_y_all)

# Use the model to make prediction the test data
test_predl = stack_te.predict(test)

# Performing inverse log transform(raise to 10) on the predictions 
test_pred =(10**test_predl) - 1
# Negative predictions for the number of likes are converted to 0.
test_pred[test_pred < 0] = 0

# Rounding the predictions
output = np.round_(test_pred)

# Creating the prediction file titled 'best_stack.csv'
sub = open('best_stack.csv','w+')
sub.write('Id,Predicted\n')
for index, prediction in zip(test_id,output):
    sub.write(str(index) + ',' + str(prediction) + '\n')
sub.close()

# The prediction file that has been created can be submitted on Kaggle to reproduce our best score.

