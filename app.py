import pandas as pd
import numpy as np
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score

s = pd.read_csv('C:/Users/travi/Box/OPAN6607/social_media_usage.csv') #load in dataset
s = s.dropna() #begin cleaning

def clean_sm(x): #clean columns in dataset. 
    x = np.where(x==1,1,0)
    return x

ss = pd.DataFrame(s) #create new empty dataframe

#create a new column that is the web1h (linkedin)
#column cleaned so that it is binary (1,0 / yes,no)
sm_li = pd.Series(clean_sm(s["web1h"].values), name = "sm_li") 

#trim the fat off original dataset to only use certain columns in analysis
#then merge to create our dataframe for analysis
s_add = s[["income", "educ2", "par", "marital", "gender", "age"]]
ss = pd.concat([sm_li, s_add], axis = 1)

#more cleaning to remove irrelevant responses
ss["income"] = np.where(ss["income"] > 9, np.nan, ss["income"])
ss["educ2"] = np.where(ss["educ2"] > 8, np.nan, ss["educ2"])
ss["par"] = np.where(ss["par"] > 2, np.nan, ss["par"])
ss["marital"] = np.where(ss["marital"] > 6, np.nan, ss["marital"])
ss["gender"] = np.where(ss["gender"] > 2, np.nan, ss["gender"]) 
ss["age"] = np.where(ss["age"] > 97, np.nan, ss["age"])
#process above converted datatypes to float64
#not sure if it is necessary, but convered them back
#to their original datatypes
ss["income"] = ss["income"].astype('Int64')
ss["educ2"] = ss["educ2"].astype('Int64')
ss["par"] = ss["par"].astype('Int64')
ss["marital"] = ss["marital"].astype('Int64')
ss["gender"] = ss["gender"].astype('Int64')
ss["age"] = ss["age"].astype('Int64')
#last bit of cleaning to make sure NAs were dropped
ss = ss.dropna()

y = ss["sm_li"]
X = ss[["income","educ2","par","marital","gender","age"]]

#create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X.values,
                                                   y,
                                                   stratify=y,
                                                   test_size=0.2, #80% train/20%test
                                                   random_state=141101) #setting seed for reproducability
ss = ss.dropna() #sanity check

#training our model
lr = LogisticRegression(class_weight="balanced")
lr.fit(X_train, y_train)