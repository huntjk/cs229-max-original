# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 13:27:49 2017

@author: max
"""
#from mpi4py import MPI
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle_toolbox

### IF THERE IS A PROBLEM WITH NETCDF MODULE THERE IS ALSO A SAVED PICKLE FILE TO DO THE CLUSTERING AS BACKUP
'''
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
    
def getLabels(data, centroids):
    #find nearest centroid to each point
    x,y,z=data.shape
    labels=np.zeros((x,y),int)
    
    for row in range(x):
        for col in range(y):
            elem=data[row,col]
            centroid_x=centroids[0]
            centroid_y=centroids[1]
            centroid_val=centroids[2]   
            dist_val=(elem-centroid_val)**2
            dist_X=(row-centroid_x)
            #Not perfect you would need if distance > 96   
            # min(180 - abs(row)+180 - abs (centroid_row) , )
            #dist_X=np.mod(dist_X,96)
            dist_Y=(col-centroid_y)**2
            dist_XY=dist_X**2+dist_Y
            #HOW MUCH YOU WEIGHT DISTANCE TO VALUE
            alpha=.00005
            distance= np.sqrt((dist_val).sum(axis=1))+alpha*dist_XY
            labels[row,col]=distance.argmin()
    
    #TODO create a label for land mass
    #labels=
    return labels

def getCentroids(data, labels, k):
    x,y,z=data.shape
    centroids=[np.zeros(k),np.zeros(k),np.zeros((k,z))]    
    #test=data[indx,indy].mean(axis=0)
    for centre in range(k):
        (indx,indy)=np.where(labels==centre)
        centroids[0][centre]=indx.mean()
        centroids[1][centre]=indy.mean()
        centroids[2][centre]=data[indx,indy].mean(axis=0)
        
        #[indx,indy,centroids[centre]]=[indx,indy,data[indx,indy].mean(axis=0)]
    return centroids

def getRandomCentroids(data, k):
#    k=CLUSTER_NUM
    X,Y,Z = data.shape
    centroids_X = np.random.choice(X, k)
    centroids_Y = np.random.choice(Y, k)
    centroids=[centroids_X,centroids_Y,data[centroids_X,centroids_Y]]
    return centroids

def replaceWithCentroids(A,labels,centroids,k):
    for center in range(k):
        (indx,indy)=np.where(labels==center)
        A[indx,indy]=centroids[2][center]
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
    CLUSTER_NUM = 40;
    MAX_ITERATIONS = 70;
    iterations = 0    
    centroids=getRandomCentroids(data, CLUSTER_NUM)
    
    while iterations < MAX_ITERATIONS:
        iterations += 1
        
        # Labels to each datapoint
        labels = getLabels(data, centroids)
        centroids = getCentroids(data, labels, CLUSTER_NUM) 
        
    B=replaceWithCentroids(data,labels,centroids,CLUSTER_NUM) 
    pickle_toolbox.plot3d(data_dic,B[:,:,1])  
    pickle_toolbox.plot3dcolormesh(data_dic,B[:,:,1])
    pickle_toolbox.plot3dcolormesh(data_dic,B[:,:,2])
    pickle_toolbox.plot3dcolormesh(data_dic,B[:,:,3])


    return centroids[2]

def removeLand(data_dic2):
    pkl_name='lsmask.19294.nc.pkl'
    data_dic_lsmask = pickle_toolbox.opendict(pkl_name)
    pickle_toolbox.plot3dcolormesh(data_dic_lsmask,np.transpose(data_dic_lsmask['lsmask'][0]))
    
    temp_matrix=data_dic2["skt"]
    select_ocean=data_dic_lsmask['lsmask']+1
    for time,elem in enumerate(temp_matrix):
        temp_matrix[time]=np.multiply(elem,select_ocean)
    
    data_dic2['skt']=temp_matrix
    return data_dic2

def main():
    
    #Read in data and visualize  
    pkl_name2='skt2.sfc.mon.mean.nc.pkl'
    data_dic2 = pickle_toolbox.opendict(pkl_name2)
    pickle_toolbox.plot3dcolormesh(data_dic2,np.transpose(data_dic2['skt'][1]))
    
    data_dic=removeLand(data_dic2)
    
    pickle_toolbox.plot3dcolormesh(data_dic,np.transpose(data_dic['skt'][1]))

    
    data=np.transpose(data_dic['skt'])
    data=preprocessdata(data)
    pickle_toolbox.plot3d(data_dic,data[:,:,1])
    
    #Runs our K means algorythm and returns 
    Kcluster=Kmeans(data,data_dic)
    Kcluster=np.transpose(Kcluster)
    
    #save as a csv file, note that it is saving a detrended value of centroids
    pd.DataFrame(Kcluster,index=data_dic["time"]).to_csv("KMeansCluster.csv")    
    return 1

if __name__ == '__main__':
    main()
