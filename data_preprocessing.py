# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 13:27:49 2017

@author: max
"""
#from mpi4py import MPI
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.io.shapereader as shapereader
import matplotlib.pyplot as plt

'''
### IF THERE IS A PROBLEM WITH NETCDF MODULE THERE IS ALSO A SAVED PICKLE FILE TO DO THE CLUSTERING AS BACKUP

from netCDF4 import Dataset, num2date

#https://www.esrl.noaa.gov/psd/repository/entry/show?entryid=synth%3Ae570c8f9-ec09-4e89-93b4-babd5651e7a9%3AL25jZXAucmVhbmFseXNpczIuZGVyaXZlZC9nYXVzc2lhbl9ncmlkL3NrdC5zZmMubW9uLm1lYW4ubmM%3D
nc_filename = 'skt.sfc.mon.mean.nc'
netc = nc.Dataset(nc_filename)
print(nc.variables)
h = nc.variables["skt"]
print(h[1,1,1])

times = nc.variables['time']
jd = num2date(times[:],times.units)
hs = xr.DataArray(h)
print(hs)

lat = nc.variables['lat'][:]
lon = nc.variables['lon'][:]
time = nc.variables['time']
'''

import pickle_toolbox

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


#visualization code modified from https://matplotlib.org/examples/mplot3d/surface3d_demo3.html
def plot3d(data_dic,skt):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    # Make data.
    X = data_dic['lat']
    Y = data_dic['lon']
    X, Y = np.meshgrid(X, Y)
    Z = skt
    
    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    
    # Customize the z axis.
    #ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    return plt.show()

#visualization code for a color mesh (you can see the continents)
def plot3dcolormesh(data_dic,skt):
    fig = plt.figure()
    
    # Make data.
    X = data_dic['lat']
    Y = data_dic['lon']
    X, Y = np.meshgrid(X, Y)
    Z = skt
    
    # Plot the surface.
    plt.pcolormesh(X, Y, Z)
    
    # Customize the z axis.
    #ax.set_zlim(-1.01, 1.01)
    #ax.zaxis.set_major_locator(LinearLocator(10))
    #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    
    # Add a color bar which maps values to colors.
    #fig.colorbar(surf, shrink=0.5, aspect=5)
    
    return plt.show()
    
def getLabels(data, centroids):
    #find nearest centroid to each point
    x,y,z=data.shape
    labels=np.zeros((x,y),int)
    for row in range(x):
        for col in range(y):
            elem=data[row,col]
            distance= np.sqrt(((elem-centroids)**2).sum(axis=1))
            labels[row,col]=distance.argmin()
    return labels

def getCentroids(data, labels, k):
    x,y,z=data.shape
    centroids=np.zeros((k,z))    
    #test=data[indx,indy].mean(axis=0)
    for center in range(k):
        (indx,indy)=np.where(labels==center)
        centroids[center]=data[indx,indy].mean(axis=0)
    return centroids

def getRandomCentroids(data, k):
    X,Y,Z = data.shape
    centroids_X = np.random.choice(X, k)
    centroids_Y = np.random.choice(Y, k)
    centroids=data[centroids_X,centroids_Y]
    return centroids

def replaceWithCentroids(A,labels,centroids,k):
    for center in range(k):
        (indx,indy)=np.where(labels==center)
        A[indx,indy]=centroids[center]
    return A

#Get anomaly 
def preprocessdata(data):
    x,y,z= data.shape
    means=np.zeros((x,y))
    for row in range(x):
        for col in range(y):
            for monthi in range(12):
                month=range(monthi,z,12)
                means=data[row,col,month].mean()
                data[row,col,month]=data[row,col,month]-means
    return data

def Kmeans(data,data_dic):
    CLUSTER_NUM = 16;
    MAX_ITERATIONS = 50;
    iterations = 0
    
    centroids=getRandomCentroids(data, CLUSTER_NUM)
    
    while iterations < MAX_ITERATIONS:
        iterations += 1
        
        # Labels to each datapoint
        labels = getLabels(data, centroids)
        centroids = getCentroids(data, labels, CLUSTER_NUM) 
        
    B=replaceWithCentroids(data,labels,centroids,CLUSTER_NUM) 
    plot3d(data_dic,B[:,:,1])    
    return centroids

def main():
    
    #Read in data and visualize
    pkl_name='skt.sfc.mon.mean.nc.pkl'
    data_dic = pickle_toolbox.opendict(pkl_name)
    plot3dcolormesh(data_dic,np.transpose(data_dic['skt'][1]))
    data=np.transpose(data_dic['skt'])
    data=preprocessdata(data)
    plot3d(data_dic,data[:,:,1])
    
    #Runs our K means algorythm and returns 
    Kcluster=Kmeans(data,data_dic)
    Kcluster=np.transpose(Kcluster)
    
    #save as a csv file, note that it is saving a detrended value of centroids
    pd.DataFrame(Kcluster,index=data_dic["time"]).to_csv("KMeansCluster.csv")    
    return 1

if __name__ == '__main__':
    main()
