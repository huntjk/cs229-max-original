# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 07:00:04 2017

@author: max
"""

import numpy as np
import pandas as pd 
import baseline
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
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVR
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import statsmodels.api as sm




#Naive KMeans Clustering, alpha being the tradeoff between value and spacial clustering
def readInCluster(lag_months,alpha,name):   
    K_data=pd.read_csv(name)    
    K_data["date"]=pd.to_datetime(K_data.iloc[:,0],unit="h",origin="1800-01-01")
    K_data=K_data.set_index(K_data["date"])
    K_data=K_data.iloc[:,1:-1]
    return K_data

#Naive KMeans Clustering, alpha being the tradeoff between value and spacial clustering
def readInKMeans(lag_months=0,alpha=0):   
    K_data=pd.read_csv('KMeansCluster.csv')    
    K_data["date"]=pd.to_datetime(K_data.iloc[:,0],unit="h",origin="1800-01-01")
    K_data=K_data.set_index(K_data["date"])
    K_data=K_data.iloc[:,1:-1]
    return K_data

#Naive KMeans Clustering, alpha being the tradeoff between value and spacial clustering
def readInRMeans(n,lag_months=0,alpha=0):   
    K_data=pd.read_csv('RSquare_KMeansCluster_70_a1.csv')    
    #K_data=pd.read_csv('RSquare_KMeansCluster.csv')   
    K_data["date"]=pd.to_datetime(K_data.iloc[:,0],unit="h",origin="1800-01-01")
    K_data=K_data.set_index(K_data["date"])
    K_data=K_data.iloc[:,1:-1]
    K_data=K_data[K_data.columns[::-1]]
    #K_data=K_data.iloc[:,-n]
    return K_data

#Naive KMeans Clustering, alpha being the tradeoff between value and spacial clustering
def readInPCA(n):      
    K_data=pd.read_csv('PCACluster_600.csv')    
    K_data["date"]=pd.to_datetime(K_data.iloc[:,0],unit="h",origin="1800-01-01")
    K_data=K_data.set_index(K_data["date"])
    K_data=K_data.iloc[:,1:-1]
    K_data=K_data.iloc[:,:n]
    return K_data



def runSklearnModels(X,Y,data_vec_results):
    test_fraction = 0.3
    seed = 7 
    X_train, X_validation, Y_train, Y_validation = train_test_split(X,Y, test_size=test_fraction, random_state=seed,shuffle=False)
    
    
    #select models to evaluate
    models = []
    models.append(('LinR', LinearRegression()))
    models.append(('SVM', SVR()))
    models.append(('Ridge', Ridge(alpha=.5)))
   # models.append(('Lasso', Lasso(alpha=.5)))
   # models.append(('Multi-layer Perceptron regressor', MLPRegressor(hidden_layer_sizes=(40, ))))
    #models.append(('ElasticNet', ElasticNet(alpha=.5,l1_ratio=0.5)))
    
    #model=LogisticRegression()
    # evaluate each model in turn
    results = []
    names = []
    savedmodel=[]
    
    for name, model in models:
        kfold = model_selection.TimeSeriesSplit(n_splits=3)
        fitted=model.fit(X_train, Y_train)
        cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold,scoring='neg_mean_squared_error')
        results.append(cv_results)
        savedmodel.append(fitted)
        names.append(name)
        msg = "Time Series CV mean (n=3) %s: %f (std %f)" % (name, -cv_results.mean(), cv_results.std())
        print(msg)
       # print('Coefficients: \n', [X.columns,model.coef_])
        print('Train Set R^2:  ',model.score(X_train,Y_train))
        print('Validation R^2:  ',model.score(X_validation,Y_validation))
        if (name=='Ridge'):
            #data_vec_results.append(model.score(X_validation,Y_validation))
            data_vec_results.append(cv_results.mean())
                        
                        
    #X2 = sm.add_constant(X)
    #est = sm.OLS(Y, X2)
    #est2 = est.fit()
    #print(est2.summary())
    
    return data_vec_results

def PCAhyperparam(data_dic,var): 
    #lag_months=[0,1,2,3,6,8,10,12,14,16]
    lag_months=[0,1,2,4,6,9,12]
    for lag in lag_months:
        #Peru   lat 0 to -7   lon 283-288  -> Peru is centered around -5, -75 
        name='Peru'
        Peru_data_temp=readInPredictandLatLon(data_dic,data_dic[var],lat1=-7,lat2=-3,lon1=283,lon2=288,lag_month=lag,name=name)    
        site_data=Peru_data_temp
        #n=[5,10,15,20,30,40,50,60,70,80,90,100,110,120,140,180,200,250,300,350,400,450,500,550,600]
        n=[1,5,10,20,30,40,50,60,70,80,90,100,110,120,130,140]
        data_vec=[]
        data_vec_results=[]
        for num in n:
            readin=readInPCA(num)
            datatemp=mergeDatasets(site_data,readin)
            data_vec.append(datatemp)
            
        for index,dataf in enumerate(data_vec):
                   # print("-------------------------")
                   # print(n[index])
                    
                    X = dataf.iloc[:,1:]
                    Y = dataf.iloc[:,0]
                    data_vec_results=runSklearnModels(X,Y,data_vec_results)
                            
        plt.plot(n,data_vec_results)
        plt.ylabel('Cross Validation Score')
        plt.xlabel('Number of Clusters')
        plt.title('PCA HyperParameter Analysis - Lag %d months'%lag)
        plt.show()

    return 1 

def MonthLagsAnalysis(lag_month,site_datas,tele_data,PCAnum,data_dic_skt):
    #Constants
    data_vec_results=[]
    #Reading in the predictands we want to evaluate the temperature of Peru at 0,1,6,12 months
    for lag_months in lag_month:
        print("-------------------------")
        print("LAG MONTH --- %d "%lag_months)
        print("-------------------------")
        

            
        KMeans_data=readInKMeans(lag_months)
        PCA_data=readInPCA(PCAnum)
        Rmeans_data=readInRMeans(70,lag_months)
        NoCluster_data=readInCluster(70,lag_months,'NoCluster.csv')
        
        for site in site_datas:
            print('-----------------------')
            print(site)
        
            if "Peru" in site:
                #Peru   lat 0 to -7   lon 283-288  -> Peru is centered around -5, -75 
                Peru_data_temp=readInPredictandLatLon(data_dic_skt,data_dic_skt['skt'],lat1=-7,lat2=-3,lon1=283,lon2=288,lag_month=lag_months,name='Peru')    
                #Peru_data_temp.to_csv('Peru_data_temp_lag(%d).csv' % lag_months)
                site_name={'Peru':Peru_data_temp}
            
            if "California" in site:
                #California is centered Lat 34 to 36 and -120 to -117
                Cal_data_temp=readInPredictandLatLon(data_dic_skt,data_dic_skt['skt'],lat1=34,lat2=36,lon1=360-120,lon2=360-117,lag_month=lag_months,name='California')
                #Cal_data_temp.to_csv('Cal_data_temp_lag(%d).csv' % lag_months)
                #Right now we are working with a single site, site data could easily be a vector of sites            
                site_name['California']=Cal_data_temp
            
            site_data=site_name[site]           
            
            datat=mergeDatasets(site_data,tele_data) #makes sure dates match too
            datak=mergeDatasets(site_data,KMeans_data)
            data_pca=mergeDatasets(site_data,PCA_data)
            datar=mergeDatasets(site_data,Rmeans_data)
            dataname = {0 : "Teleconection", 1 : "KMeans Centroids",2:"PCA Decomposition",3:"RSquare-Kmeans",4:"No Cluster Thetas"}
            
            data_vec=[datat,datak,data_pca,datar,datanocluster]
            #dataname = {0 : "Teleconection"}
            #data_vec=[datat]
            for index,dataf in enumerate(data_vec):
                print("-------------------------")
                print(dataname[index])
                
                X = dataf.iloc[:,1:]
                Y = dataf.iloc[:,0]
                data_vec_results=runSklearnModels(X,Y,data_vec_results)
            
    return True

def main():
    
    #Vector of sites to be analyzed
    #site_datas=['Peru','California']
    site_datas=['Peru']
    '''
    ## READ IN FROM NCAR Skin Temperature
    '''   
    var='skt'
    pkl_name='skt2.sfc.mon.mean.nc.pkl'
    data_dic = p_t.opendict(pkl_name)   
    #p_t.plot3dcolormesh(data_dic_skt,data_dic_skt['skt'][0])
    
    #if PCAhyperparam(data_dic,var):
    #     print('PCA number of eigenvectors analysis complete')
         
    lag_month=[1]
    PCAnum=70
    if MonthLagsAnalysis(lag_month,site_datas,tele_data,PCAnum,data_dic):
        print('Monthly Lags Analysis Complete')
    
    
    
    return 1

if __name__ == '__main__':
    main()