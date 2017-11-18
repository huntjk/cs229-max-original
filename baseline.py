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
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVR
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

def readInTeleData():
    tele_data=pd.read_csv('teleconnections.csv',skiprows=[1])
    
    tele_data["date"]=pd.to_datetime(tele_data['yyyy'].astype(str) + '/' + tele_data['mm'].astype(str), format = "%Y/%m")
    tele_data=tele_data.replace(-99.9,np.NaN)
    tele_data=tele_data.iloc[:,2:]
    tele_data=tele_data.drop('TNH', 1)
    tele_data=tele_data.drop('PT', 1)
    
    #detrend 
    df=tele_data
    means=df.groupby(df["date"].dt.month).mean()
    new=df.iloc[:,:-1]
    for elem in range(12):
        new.loc[(df["date"].dt.month==elem),:]=df.where(df["date"].dt.month==elem).iloc[:,:-1]-means.iloc[elem-1]
    new=new.set_index(df.iloc[:,-1])    
    return new

def readInPredictands():
    #https://www7.ncdc.noaa.gov/CDO/CDODivisionalSelect.jsp
    site1_data=pd.read_csv('california.txt')
    
    site1_data["date"]=pd.to_datetime(site1_data['\tYear\t'].astype(str) + '/' + site1_data['\tMonth'].astype(str), format = "%Y/%m")
    site1_data=site1_data[["date","\tPCP"]]
    site1_data=site1_data.fillna(site1_data.mean())
    
    #De_trend
    df=site1_data
    means=df.groupby(df["date"].dt.month).mean()
    new=df.iloc[:,1:]
    for elem in range(12):
        new.loc[(df["date"].dt.month==elem),:]=df.where(df["date"].dt.month==elem).iloc[:,1:]-means.iloc[elem-1]
    new=new.set_index(df.iloc[:,0])
    
    return new

def mergeDatasets(site_data,tele_data,):
    df=pd.concat([site_data, tele_data] , axis=1, join='inner')     
    #Select only y we want to predict
    df=df.loc[-pd.isnull(df['\tPCP']),:]
    
    #questionable method to deal with NA, maybe impute would be better?
    df=df.fillna(method='pad')
    
    return df

def readInKMeans():
    K_data=pd.read_csv('KMeansCluster.csv')
    
    K_data["date"]=pd.to_datetime(K_data.iloc[:,0],unit="h",origin="1800-01-01")
    K_data=K_data.set_index(K_data["date"])
    K_data=K_data.iloc[:,1:-1]
    return K_data


def main():
    #Constants
    test_fraction = 0.3
    seed = 7
    
    tele_data=readInTeleData()
    site_data=readInPredictands()
    KMeans_data=readInKMeans()
    
    datat=mergeDatasets(site_data,tele_data) #makes sure dates match too
    datak=mergeDatasets(site_data,KMeans_data)
    dataname = {0 : "Teleconection", 1 : "KMeans Centroids"}
    
    data_vec=[datat,datak]
    for index,elem in enumerate(data_vec):
        print("-------------------------")
        print(dataname[index])
        dataf=elem
        X = dataf.iloc[:,1:]
        Y = dataf.iloc[:,0]
        X_train, X_validation, Y_train, Y_validation = train_test_split(X,Y, test_size=test_fraction, random_state=seed)
        
        
        #select models to evaluate
        models = []
        models.append(('LinR', LinearRegression()))
        models.append(('SVM', SVR()))
        models.append(('Ridge', Ridge(alpha=1e-1)))
        
        model=LogisticRegression()
        # evaluate each model in turn
        results = []
        names = []
        savedmodel=[]
        
        for name, model in models:
            kfold = model_selection.KFold(n_splits=10, random_state=seed)
            fitted=model.fit(X_train, Y_train)
            cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold)
            results.append(cv_results)
            savedmodel.append(fitted)
            names.append(name)
            msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
            print(msg)
           # print('Coefficients: \n', [X.columns,model.coef_])
            print('R^2: \n',model.score(X_validation,Y_validation))
        
    
    return 1

if __name__ == '__main__':
    main()