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
from scipy import linalg
import math
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
import Jas_R_squared as jas

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
    
def getLabels(data, centroids,alpha):
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
            dist_X=(row-centroid_x)**2
            
            #Not perfect you would need if distance > 96   
            # min(180 - abs(row)+180 - abs (centroid_row) , )
            #dist_X=np.mod(dist_X,96)
            dist_Y=(col-centroid_y)**2
            dist_XY=dist_X+dist_Y
            #HOW MUCH YOU WEIGHT DISTANCE TO VALUE
            #alpha=.00005
            distance= np.sqrt((dist_val).sum(axis=1))+alpha*dist_XY
            labels[row,col]=distance.argmin()
    
    #TODO create a label for land mass
    #labels=
    return labels

def getCentroids(data, labels, k):
    x,y,z=data.shape
    centroids=[np.zeros(k),np.zeros(k),np.zeros((k,z))]    
    #test=data[indx,indy].mean(axis=0)
    
    #for centre in np.unique(labels):
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
    row=0
    col=0
    for row in range(x):
        for col in range(y):
            for monthi in range(12):
                month=range(monthi,z,12)
                means=data[row,col,month].mean()

                #NOTE DO NOT USE -= will not work
                data[row,col,month]=data[row,col,month]-means

    return data

def Kmeans(data,data_dic,knum,alpha):
    CLUSTER_NUM = knum
    MAX_ITERATIONS = 90
    iterations = 0    
    centroids=getRandomCentroids(data, CLUSTER_NUM)
    
    while iterations < MAX_ITERATIONS:
        iterations += 1
        
        # Labels to each datapoint
        labels = getLabels(data, centroids,alpha)
        centroids = getCentroids(data, labels, CLUSTER_NUM) 
        
    #B=replaceWithCentroids(data,labels,centroids,CLUSTER_NUM) 
    #pickle_toolbox.plot3d(data_dic,B[:,:,1])  
    #pickle_toolbox.plot3dcolormesh(data_dic,B[:,:,1].T)
        if (iterations%50==40):
            print("Cluster Labels at interation %d, with cluster num=%d, alpha=%f"%(iterations,knum,alpha))
            pickle_toolbox.plot3dcolormesh(data_dic,labels.T)

    return centroids[2]

def removeLand(data_dic2,var):
    pkl_name='lsmask.19294.nc.pkl'
    data_dic_lsmask = pickle_toolbox.opendict(pkl_name)
    #pickle_toolbox.plot3dcolormesh(data_dic_lsmask,(data_dic_lsmask['lsmask'][0]))
    
    temp_matrix=data_dic2[var]
    select_ocean=data_dic_lsmask['lsmask']+1
    for time,elem in enumerate(temp_matrix):
        temp_matrix[time]=np.multiply(elem,select_ocean)
    
    data_dic2[var]=temp_matrix
    return data_dic2

def PCA(data,data_dic,n):
    lat,lon,time=data.shape
    A=data.reshape(lat*lon,time)
    
    # calculate the covariance matrix
    cov_matrix = np.cov(A, rowvar=False)
    
    # calculate eigenvectors & eigenvalues of the covariance matrix
    e_vals, e_vecs = linalg.eigh(cov_matrix)
    
    # sort eigenvalue in decreasing order
    idx = np.argsort(e_vals)[::-1]
    e_vecs = e_vecs[:,idx]
    
    # sort eigenvectors according to same index
    e_vals = e_vals[idx]
    
    # select the first n eigenvectors, which we use as centroids
    e_vecs = e_vecs[:, :n]

    return e_vecs

def runPCA(data,data_dic,var):
    
    for n in [600]:
        PCAcluster=PCA(data,data_dic,n)
        #save as a csv file, note that it is saving a detrended value of centroids
        pd.DataFrame(PCAcluster,index=data_dic["time"]).to_csv("PCACluster_%d.csv"%n) 
        PCAcluster=PCAcluster.T
        labels=getLabels(data, [0,0,PCAcluster],alpha=0)
        B=replaceWithCentroids(data,labels,[0,0,PCAcluster],n)
        print("PCA Clustering with N=%d"%n)
        pickle_toolbox.plot3dcolormesh(data_dic,labels.T)
    return True

def runKclustering(data,data_dic):
    #Runs our K means algorythm and returns 
    knum=[80]
    alpha=[.0005]
    #knum=[40,60,80]
    #alpha=[0.00005,0.0005,0.005,0.05]
    KclusteringNestedDic={}
    for k in knum:
        KclusteringNestedDic[k]= {}
        for a in alpha:
                Kcluster=Kmeans(data,data_dic,k,a)
                KclusteringNestedDic[k][a]=np.transpose(Kcluster) 
            
    pickle_toolbox.savedict(KclusteringNestedDic, path='KclusteringNestedDic')
    #save as a csv file, note that it is saving a detrended value of centroids
    #pd.DataFrame(Kcluster,index=data_dic["time"]).to_csv("KMeansCluster_70_a1.csv")    

    return 1

def readIn(name,var,rmvland=True):
        #Read in data and visualize  
    pkl_name2=name
    data_dic2 = pickle_toolbox.opendict(pkl_name2)
    #pickle_toolbox.plot3dcolormesh(data_dic2,data_dic2[var][1])
    
    data_dic=data_dic2
    
    if rmvland:
        data_dic=removeLand(data_dic2,var)
    
    trainidx=math.floor((data_dic['time'].shape[0])*.7)
    train_dic=data_dic.copy()
    train_dic['time']=data_dic['time'][:trainidx]
    train_dic[var]=data_dic[var][:trainidx,:,:]
    pickle_toolbox.savedict(train_dic,"train_%s"%var)
    pickle_toolbox.savedict(data_dic,"alldata_%s"%var)
    
    return data_dic,train_dic

def selectThetasAsCentroidNoCluster(R_Square_Matrix,data,data_dic):
    long_range,lat_range,time = data.shape 
    reshaped_theta=R_Square_Matrix.reshape(long_range*lat_range)
    reshaped_data=data.reshape((long_range*lat_range,time))
    max200val=reshaped_theta.argsort()
    reshaped_theta[max200val[-400:]]
    nocluster_centroid=np.zeros((400,time))
    nocluster_centroid=reshaped_data[max200val[-400:]]
    pd.DataFrame(nocluster_centroid.T,index=data_dic["time"]).to_csv("NoCluster.csv")    
    return 1

def runRSquare(data,data_dic):
    lag_months=1
    Peru_data=pd.read_csv("Peru_0.csv").iloc[:,1]
    #Run Linear Regression & get R^2 values 
    long_range,lat_range,time = data.shape 
    R_Square_Matrix = np.zeros((long_range,lat_range))
    for long in range(0,long_range):
        for lat in range(0,lat_range):
            X = np.transpose([data[long, lat, 0:-lag_months]])
            Y = Peru_data[lag_months:]
            linRegModel = linear_model.LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
            linRegModel.fit(X,Y)
            R_Square_Matrix[long,lat] = linRegModel.score(X,Y)
  
    if selectThetasAsCentroidNoCluster(R_Square_Matrix,data,data_dic)
        print("Selected 400 largest thetas as centroids without clustering, saved to noCluster.csv ")    
    #visualize 
    pickle_toolbox.plot3dcolormesh(data_dic, R_Square_Matrix.T)
    
    #np.max(R_Square_Matrix)
    pass_in_R_Square_Matrix = np.copy(R_Square_Matrix) #Next step overrides R_Square_Matrix 
    max_iterations = 80
    RclusteringNestedDic={}
    for k_num_clusters in [60,70,80,90]:
        RclusteringNestedDic[k_num_clusters]= {}
        for cartesian_weight_for_kmeans in [0.0005,0.005]: 
                #Runs our K means algorithm
                centroids = jas.initializeCentroids( data_dic, R_Square_Matrix, k_num_clusters)
                centroids = jas.runKMeans(data_dic, R_Square_Matrix, centroids, max_iterations, cartesian_weight_for_kmeans)
                #Replace grid points with closest centroid
                clustered_R_Square_Matrix = jas.replaceWithCentroids(data_dic, pass_in_R_Square_Matrix, centroids, cartesian_weight_for_kmeans)
                
                label=np.unique(clustered_R_Square_Matrix)
                long_range,lat_range,time = data.shape 
                labels = np.zeros((long_range,lat_range))
            
               
                for idx,elem in enumerate(label):
                    (indx,indy)=np.where(clustered_R_Square_Matrix==elem)
                    labels[indx,indy]=idx
    
                #visualize 
                pickle_toolbox.plot3dcolormesh(data_dic, labels.T)    
                k=len(label)
                x,y,z=data.shape
                centroids=[np.zeros(k),np.zeros(k),np.zeros((k,z))]    
                #test=data[indx,indy].mean(axis=0)
                
                #for centre in np.unique(labels):
                for centre in range(k):
                    (indx,indy)=np.where(labels==centre)
                    centroids[0][centre]=indx.mean()
                    centroids[1][centre]=indy.mean()
                    centroids[2][centre]=data[indx,indy].mean(axis=0)
                cluster=centroids[2].T
                RclusteringNestedDic[k_num_clusters][cartesian_weight_for_kmeans]=cluster
    #save as a csv file, note that it is saving a detrended value of centroids
    pickle_toolbox.savedict(RclusteringNestedDic, path='RclusteringNestedDic')
    
    #visualize 
    #pickle_toolbox.plot3dcolormesh(data_dic, clustered_R_Square_Matrix.T)
   
    #visualize 
    #pickle_toolbox.plot3dcolormesh(data_dic, clustered_R_Square_Matrix)
    return True

def main():
    
    # for skt skin temperature
    var='skt'
    name='skt2.sfc.mon.mean.nc.pkl'
    data_dic,train_dic=readIn(name,var,True)

    print("visual check that map looks right")
    pickle_toolbox.plot3dcolormesh(data_dic,data_dic[var][1])
    
    data=np.transpose(data_dic[var].copy())
    
    #remove monthly means
    data=preprocessdata(data) 
    print("visual check that mean removal looks right, with land removed")
    pickle_toolbox.plot3dcolormesh(data_dic,(data[:,:,1]).T)
    
    #if runPCA(data,data_dic,var):
    #    print("Principal Component Analysis Complete") 
        
    #if runKclustering(data,data_dic):
    #    print("Kclustering Complete") 
    
    if runRSquare(data,data_dic):
        print("RSquare Complete") 
    return 1

if __name__ == '__main__':
    main()
