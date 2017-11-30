# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 07:00:04 2017

@author: max
"""

import numpy as np
import pandas as pd 
import cartopy.crs as ccrs
import cartopy.io.shapereader as shapereader
import matplotlib.pyplot as plt
import pickle_toolbox as p_t
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
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
import statsmodels.api as sm

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

def readInPredictands(lag_month=1):
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
    if (lag_month==0):
        new=new.set_index(df.iloc[:,0])
        return new
    new_lag=new.iloc[lag_month:,:]
    new_lag=new_lag.set_index(df.iloc[:-lag_month,0])  
    return new_lag

def readInPredictandLatLon(data_dic,selection,lat1,lat2,lon1,lon2,lag_month=1,):
    #https://www7.ncdc.noaa.gov/CDO/CDODivisionalSelect.jsp
    lat,=np.where((data_dic['lat']>lat1) & (data_dic['lat']<lat2))
    lon,=np.where((data_dic['lon']>lon1) & (data_dic['lon']<lon2))
    selection=selection[:,lat]
    selection=selection[:,:,lon]
    selection_mean=np.mean(selection,axis=1)
    selection_mean=np.mean(selection_mean,axis=1)
    
    date_vec=pd.to_datetime(data_dic['time'],unit="h",origin="1800-01-01")
    
    if (lag_month==0):
        no_lag=pd.DataFrame(selection_mean)
        no_lag=no_lag.set_index(date_vec) 
        return no_lag

    selection_lag=selection_mean[lag_month:]
    select=pd.DataFrame(selection_lag)  
    selection_lag=select.set_index(date_vec[:-lag_month]) 
    
    return selection_lag

def mergeDatasets(site_data,tele_data,):
    df=pd.concat([site_data, tele_data] , axis=1, join='inner')     
    #Select only y we want to predict
    #df=df.loc[-pd.isnull(df['\tPCP']),:]
    
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
    lag_months=3
    tele_data=readInTeleData()
    #site_data=readInPredictands(lag_month)
    
    ## READ IN FROM NCAR
    pkl_name='prate.sfc.mon.mean.nc.pkl'
    data_dic_prep = p_t.opendict(pkl_name)   
    p_t.plot3dcolormesh(data_dic_prep,np.transpose(data_dic_prep['prate'][0]))
    
    site_data=[]
    
    #Peru   lat 0 to -7   lon 253-258
    Peru_data_precip=readInPredictandLatLon(data_dic,data_dic['prate'],lat1=-7,lat2=0,lon1=253,lon2=258,lag_month=lag_months)
    site_data.append(Peru_data_precip)
    
    #California   lat 0 to -7   lon 253-258
    Peru_data=readInPredictandLatLon(data_dic,data_dic['prate'],lat1=-7,lat2=0,lon1=253,lon2=258,lag_month=lag_months)
    site_data.append(Peru_data)
    
    pkl_name='skt2.sfc.mon.mean.nc.pkl'
    data_dic_skt = p_t.opendict(pkl_name)   
    p_t.plot3dcolormesh(data_dic_skt,np.transpose(data_dic_skt['skt'][0]))

    #Peru   lat 0 to -7   lon 253-258
    Peru_data_temp=readInPredictandLatLon(data_dic_skt,data_dic_skt['skt'],lat1=-7,lat2=0,lon1=253,lon2=258,lag_month=lag_months)
    site_data.append(Peru_data_temp)
    
    site_data=Peru_data_temp
    
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
            kfold = model_selection.KFold(n_splits=5, random_state=seed)
            fitted=model.fit(X_train, Y_train)
            cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold)
            results.append(cv_results)
            savedmodel.append(fitted)
            names.append(name)
            msg = "CV mean %s: %f (std %f)" % (name, cv_results.mean(), cv_results.std())
            print(msg)
           # print('Coefficients: \n', [X.columns,model.coef_])
            print('R^2: \n',model.score(X_validation,Y_validation))
     
        X2 = sm.add_constant(X)
        est = sm.OLS(Y, X2)
        est2 = est.fit()
        print(est2.summary())
    
    return 1

if __name__ == '__main__':
    main()