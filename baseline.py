# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 07:00:04 2017

@author: max
"""

import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#Constants
test_fraction = 0.3
seed = 7
scoring = 'accuracy'


tele_data=pd.read_csv('teleconnections.csv',skiprows=[1])

tele_data["date"]=pd.to_datetime(tele_data['yyyy'].astype(str) + '/' + tele_data['mm'].astype(str), format = "%Y/%m")
tele_data=tele_data.replace(-99.9,np.NaN)
tele_data=tele_data.iloc[:,2:]

#https://www7.ncdc.noaa.gov/CDO/CDODivisionalSelect.jsp
site1_data=pd.read_csv('california.txt')

site1_data["date"]=pd.to_datetime(site1_data['\tYear\t'].astype(str) + '/' + site1_data['\tMonth'].astype(str), format = "%Y/%m")
site1_data=site1_data[["date","\tPCP"]]
df=pd.merge(site1_data, tele_data, on='date', how='outer')
df=df.drop('TNH', 1)
df=df.drop('PT', 1)

#
df=df.fillna(df.mean())

means=df.groupby(df["date"].dt.month).mean()

new=df.iloc[:,1:]
for elem in range(12):
    new.loc[(df["date"].dt.month==elem),:]=df.where(df["date"].dt.month==elem).iloc[:,1:]-means.iloc[elem-1]

new.to_csv("climateai_interview_data.csv")

X = new.iloc[:,1:]
Y = new.iloc[:,1]
X_train, X_validation, Y_train, Y_validation = train_test_split(X,Y, test_size=test_fraction, random_state=seed)

