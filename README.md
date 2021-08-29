# Data Science Competition- Social Media Prediction


The main task for the data challenge is to predict the number of 'likes' for given details about the simulated profiles of users on social media.

We have been provided with various attributes about the visual aspects of the users' social media profiles, the basic details, the profile pictures used in the simulation. We use this information to predict how many 'likes' the users could've likely received. Our result ranked **fourth out of 31 teams**.


### Team Name - Student Baseline

presentation slide can be found [here](https://docs.google.com/presentation/d/1HfolpTci-lAh_MLWbS5ZTfUYm19PZh1uEr2Z5sGU5Yg/edit#slide=id.gb0fe54fd99_0_43).

### Team Members:

Balaji Balasubramanian

Patcharin Cheng 

Arjun Vaithilingam Sudhakar


### Prerequisite:

1.  Make sure the dataset is linked under the following directory/load the competition dataset in Kaggle,

/kaggle/input/ift6758-a20/...
or, The data path file has been written to run it on Kaggle notebooks, rename the path file if required.

2. The codes require pandas, numpy, sklearn, xgboost, mlxtend-0.18.0, os, random, and warnings libraries to run.

Please update mlxtend to the latest version(0.18.0) before running it.(Kaggle notebook uses mlxtend-0.18.0 by default, hence no need to update here).

### Steps to run the program:

Step 1: change input directories to point at your files

Step 2: run `python script.py`

Step 3: A csv file will be generated (model prediction on the test data after training)

Step 4: Upload the csv file created under Submit Prediction in Kaggle to get the accuracy score to estimate the model generalized performance

### This notebook will generate the score as following:

Public Score = 1.70724

Private Score = 1.59018
