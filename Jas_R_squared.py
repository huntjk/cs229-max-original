#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS229 - Project 
By Jasdeep Singh & Max Evans
Variable:        Skin temperature (De-Trended)
Data to cluster: R^2 values for each individual data point for linear Regression, Cartesian Distance  
Clustering alg:  K-Means, K-medoids 
Prediction time scale: Variable (lag-months = # month into future)
Prediction location: Peru (lat: 0 to -7 , lon: 253-258)
In this file we use the R^2 values from Linear regresion to find the lat&long locations
that are the most predictive for a certain location Y. We then cluster
these locations on the globe that are close to each other and have similiar
predictive power (R^2 values). 
"""



import numpy as np
import pandas as pd
import pickle_toolbox_old as p_t
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from sklearn import linear_model
from sklearn.linear_model import LinearRegression



"""
Function: readInPredictandLatLon
This function averages the data values in a given latitude range and 
longitude range (basically avergaes over a square of points). 
This should have already been done in another program. 
"""
def readInPredictandLatLon(data_dic,selection,lat1,lat2,lon1,lon2):
    #https://www7.ncdc.noaa.gov/CDO/CDODivisionalSelect.jsp
    lat,=np.where((data_dic['lat']>lat1) & (data_dic['lat']<lat2))
    lon,=np.where((data_dic['lon']>lon1) & (data_dic['lon']<lon2))
    selection=selection[:,lat]
    selection=selection[:,:,lon]
    selection_mean=np.mean(selection,axis=1)
    selection_mean=np.mean(selection_mean,axis=1)
    return selection_mean



"""
Function: preprocessdata
Get anomaly, Subtract out the means, Detrend data by month
"""
def preprocessdata(data):
    x,y,z = data.shape #long, lat, time 
    means = np.zeros((x,y))
    for row in range(x):
        for col in range(y):
            for monthi in range(12):
                month = range(monthi,z,12) #gets a range object of all monthi's over the 60 years
                means = data[row,col,month].mean()
                data[row,col,month] = data[row,col,month]-means
    return data



"""
Function: replaceWithCentroids
Go through the grid(globe) and replace the theta values [any time compressed values] 
of each point with those of the closest centroids value. 
"""
def replaceWithCentroids(data_dic, theta_Matrix, centroids, cartesian_weight_for_kmeans):
    clustered_theta_matrix = theta_Matrix
    longitude_range,latidude_range = theta_Matrix.shape
    for i in range(0, longitude_range):
        for j in range(0, latidude_range):
            x = [data_dic['lon'][i], data_dic['lat'][j], theta_Matrix[i,j]]   
            cartesian_distance = np.linalg.norm(centroids[:,0:2] - x[0:2], axis = 1)
            theta_difference = np.sqrt(np.square(centroids[:,2] - x[2])) 
            total_differnce = theta_difference + (cartesian_weight_for_kmeans*cartesian_distance)
            index = np.argmin(total_differnce) #closest centroid will have smallest distance 
            clustered_theta_matrix[i,j] = centroids[index,2] #closest centroid's theta value
    return clustered_theta_matrix



"""
Function: runKMeans
This Function runs the K-Means algorithem on a set of provided centroids.
E step: It first sorts the data into groups that are closest to a praticular centroid 
M step: It then sets the centroids value to the average value of that cluster of points 
"""
def runKMeans (data_dic, theta_Matrix, centroids, max_iterations, cartesian_weight_for_kmeans):
    longitude_range,latidude_range = theta_Matrix.shape
    for num in range(0, max_iterations):
        clustersOfData = [[] for _ in range(centroids.shape[0])]    
        #E-Step
        for i in range(0, longitude_range):
            for j in range(0, latidude_range):
                x = [data_dic['lon'][i], data_dic['lat'][j], theta_Matrix[i,j]]   
                cartesian_distance = np.linalg.norm(centroids[:,0:2] - x[0:2], axis = 1)
                theta_difference = np.sqrt(np.square(centroids[:,2] - x[2]))               
                total_differnce = theta_difference + (cartesian_weight_for_kmeans*cartesian_distance) 
                index = np.argmin(total_differnce) #closest centroid will have smallest distance 
                clustersOfData[index].append(x)           
        #M-Step
        for i in range(0, centroids.shape[0]):
            centroids[i] = np.mean(clustersOfData[i],axis = 0)
            # This is where k-mediods differs 
    return centroids



"""
Function: runKMediods
***** This is supposed to use the L1 norm !!!!
"""
def runKMedoids (data_dic, theta_Matrix, centroids, max_iterations, cartesian_weight_for_kmeans):
    max_iterations = 1
    longitude_range,latidude_range = theta_Matrix.shape
    for num in range(0, max_iterations):
        clustersOfData = [[] for _ in range(centroids.shape[0])]    
        #E-Step
        for i in range(0, longitude_range):
            for j in range(0, latidude_range):
                x = [data_dic['lon'][i], data_dic['lat'][j], theta_Matrix[i,j]]   
                cartesian_distance = np.linalg.norm(centroids[:,0:2] - x[0:2], axis = 1)
                theta_difference = np.sqrt(np.square(centroids[:,2] - x[2])) 
                total_differnce = theta_difference + (cartesian_weight_for_kmeans*cartesian_distance)
                index = np.argmin(total_differnce) #closest centroid will have smallest distance 
                clustersOfData[index].append(x)           
        #M-Step
        for i in range(0, centroids.shape[0]):
            centroids[i] = np.mean(clustersOfData[i],axis = 0)
            # This is where k-mediods differs, replace the centroids[i] with the closest data point to it
            #(np.transpose(np.transpose(clustersOfData[i][:4])[0:2])) Converts list to array
            #For some reason asarray doesn't work 
            cartesian_distance = np.linalg.norm(np.transpose( np.transpose(clustersOfData[i][:]) [0:2]) - centroids[i][0:2], axis = 1)
            theta_difference = np.sqrt(np.square(np.transpose( np.transpose(clustersOfData[i][:]) [2]) - centroids[i,2]))
            total_differnce = theta_difference + (cartesian_weight_for_kmeans*cartesian_distance)
            index = np.argmin(total_differnce) #index of the closest data point in clustersOfData[i,:] to the centroid 
            centroids[i] = clustersOfData[i][index]
    return centroids



"""
Function: initializeCentroids
Initalizes the centroids as (long, lat, theta). 
It picks current data points as the initial centroids
"""
def initializeCentroids (data_dic, theta_Matrix, k): 
    centroids = np.zeros((k,3))
    longitude_range,latidude_range = theta_Matrix.shape
    for i in range(0,k):
        x = np.random.randint(0,longitude_range)
        y = np.random.randint(0,latidude_range)
        centroids[i] = [data_dic['lon'][x], data_dic['lat'][y], theta_Matrix[x][y]]
    return centroids




"""
Function: createTrainingDataMatrix
flattens data longitudinally 
X is matrix m = months, n = all measurements around globe 
"""
def createTrainingDataMatrix(data, lag_months):
    long,lat,time = data.shape 
    X = np.zeros((time-lag_months, long*lat)) 
    for i in range(0, time-lag_months): #last month not included, we are predicting the next month
        X[i] = data[:,:,i].flatten(order='C') #flatten row-wise
    X = np.insert(X,0,1,axis = 1) #Add column of 1's for intercept 
    return X
  


"""
Function: getExpectedValues
This Lags the expected data (Y) according to lag_months
"""
def getExpectedValues(data, lag_months):   
    Y = data[lag_months:] #first lag-months not included
    return Y



"""
Funcction: main 
"""
def main():
    #constants
    data_set = 'skt.sfc.mon.mean.nc.pkl'
    k_num_clusters = 70
    max_iterations = 150
    cartesian_weight_for_kmeans = 0.000001
    lag_months = 1
    
    #Read in data
    pkl_name = data_set
    data_dic = p_t.opendict(pkl_name) #data_dic is (lat,long,time, valuesOfSkinTemp[latitude, lonvitude, time])
    data = np.transpose(data_dic['skt']) #data is a 3d matrix of the skinTemp values 
    data = preprocessdata(data) #detrend the data  
    #Get data for the peru region:   lat 0 to -7   lon 253-258
    Peru_data = readInPredictandLatLon(data_dic, np.transpose(data),lat1=-7,lat2=0,lon1=253,lon2=258)
    
    #Run Linear Regression & get R^2 values 
    long_range,lat_range,time = data.shape 
    R_Square_Matrix = np.zeros((long_range,lat_range))
    for long in range(0,long_range):
        for lat in range(0,lat_range):
            X = np.transpose([data[long, lat, 0:-lag_months]])
            Y = getExpectedValues(Peru_data, lag_months)
            linRegModel = linear_model.LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
            linRegModel.fit(X,Y)
            R_Square_Matrix[long,lat] = linRegModel.score(X,Y)
  
    #visualize 
    p_t.plot3dcolormesh(data_dic, data[:,:,0])
    p_t.plot3dcolormesh(data_dic, R_Square_Matrix)
    
    
    pass_in_R_Square_Matrix = np.copy(R_Square_Matrix) #Next step overrides R_Square_Matrix 
    #Runs our K means algorithm
    centroids = initializeCentroids( data_dic, R_Square_Matrix, k_num_clusters)
    centroids = runKMeans(data_dic, R_Square_Matrix, centroids, max_iterations, cartesian_weight_for_kmeans)
    #Replace grid points with closest centroid
    clustered_R_Square_Matrix = replaceWithCentroids(data_dic, pass_in_R_Square_Matrix, centroids, cartesian_weight_for_kmeans)
    #visualize 
    p_t.plot3dcolormesh(data_dic, clustered_R_Square_Matrix)
    
    
    pass_in_R_Square_Matrix = np.copy(R_Square_Matrix) #Next step overrides R_Square_Matrix 
    #Run K Mediods algorithm 
    medoids = initializeCentroids( data_dic, R_Square_Matrix, k_num_clusters)
    medoids = runKMedoids(data_dic, R_Square_Matrix, medoids, max_iterations, cartesian_weight_for_kmeans)
    #Replace grid points with closest medoid
    clustered_R_Square_Matrix_medoids = replaceWithCentroids(data_dic, pass_in_R_Square_Matrix, medoids, cartesian_weight_for_kmeans)
    #visualize 
    p_t.plot3dcolormesh(data_dic, clustered_R_Square_Matrix_medoids)
    
   
    #save as a csv file, note that it is saving a detrended value of centroids
    #pd.DataFrame(Kcluster,index=data_dic["time"]).to_csv("KMeansCluster.csv") 
    
    return 1



if __name__ == '__main__':
    main()
