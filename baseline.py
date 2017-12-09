# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 14:56:11 2017

@author: max
"""

import main_analysis as mainfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
import pickle_toolbox as p_t
from sklearn.linear_model import LinearRegression


'''
# This function reads in Classical Teleconnection indexes from NOAA
# source ftp://ftp.cpc.ncep.noaa.gov/wd52dg/data/indices/tele_index.nh

column 3: North Atlantic Oscillation (NAO)
column 4: East Atlantic Pattern (EA)
column 5: West Pacific Pattern (WP)
column 6: EastPacific/ North Pacific Pattern (EP/NP)
column 7: Pacific/ North American Pattern (PNA)
column 8: East Atlantic/West Russia Pattern (EA/WR)
column 9: Scandinavia Pattern (SCA)
column 10: Tropical/ Northern Hemisphere Pattern (TNH)
column 11: Polar/ Eurasia Pattern (POL)
column 12: Pacific Transition Pattern (PT)
column 13: Explained Variance (%) of leading 10 modes
'''

def readInTeleData():
    tele_data=pd.read_csv('teleconnections.csv',skiprows=[1])
    
    tele_data["date"]=pd.to_datetime(tele_data['yyyy'].astype(str) + '/' + tele_data['mm'].astype(str), format = "%Y/%m")
    tele_data=tele_data.replace(-99.9,np.NaN)
    tele_data=tele_data.iloc[:,2:]
    tele_data=tele_data.drop('TNH', 1)
    tele_data=tele_data.drop('PT', 1)
    
    #detrend by removing monthly means
    df=tele_data
    new=df.iloc[:,:-2]
    
    #Detrend TeleData
    means=df.groupby(df["date"].dt.month).mean()
    for elem in range(12):
        new.loc[(df["date"].dt.month==elem),:]=df.where(df["date"].dt.month==elem).iloc[:,:-1]-means.iloc[elem-1]
    new=new.set_index(df.iloc[:,-1])    
    return new


#function to read in a lat lon temperature at a given lag
def readInPredictandLatLon(data_dic,selection,lat1,lat2,lon1,lon2,name,var):
    #https://www7.ncdc.noaa.gov/CDO/CDODivisionalSelect.jsp
    selection=data_dic['skt']
    lat,=np.where((data_dic['lat']>lat1) & (data_dic['lat']<lat2))
    lon,=np.where((data_dic['lon']>lon1) & (data_dic['lon']<lon2))
    selection=selection[:,lat]
    selection=selection[:,:,lon]
    selection_mean=np.mean(selection,axis=1)
    selection_mean=np.mean(selection_mean,axis=1)


    date_vec=pd.to_datetime(data_dic['time'],unit="h",origin="1800-01-01")

    print("Before Removing monthly seasonality")
    plt.plot(date_vec,selection_mean)
    plt.show()
    autocorrelation_plot(selection_mean)
    plt.show()
        


    #De trend
    for monthi in range(12):
        month=range(monthi,selection_mean.shape[0],12)
        means=selection_mean[month].mean()
        stdev=selection_mean[month].std()
        selection_mean[month]=(selection_mean[month]-means)/stdev

    no_lag=pd.DataFrame(selection_mean)
    no_lag=no_lag.set_index(date_vec) 
    
    
    print("After Removing monthly seasonality (ie - monthly mean)")
    
    plt.plot(date_vec,selection_mean)
    plt.show()

    autocorrelation_plot(no_lag)
    plt.show()
   # X=range(no_lag.shape)
    model = LinearRegression()
    X=data_dic['time'].reshape(data_dic['time'].shape[0],1)
    mymodel=model.fit(X, no_lag)
    # calculate trend
    trend = mymodel.predict(X)
    # plot trend
    #plt.plot(no_lag)
    #plt.plot(trend)
    #plt.show()
    # detrend
    detrended=(no_lag-trend)
 #   detrended = [y[i]-trend[i] for i in range(0, len(series))]
    # plot detrended
    plt.plot(detrended)
    plt.title("Detrended Values")
    plt.show()
    
    no_lag.to_csv('%s_%s_0.csv'%(name,var))
    return no_lag

def main():
    #Classical Teleconnections Data
    tele_data=readInTeleData()
    
    var='skt'
    pkl_name='skt2.sfc.mon.mean.nc.pkl'
    data_dic = p_t.opendict(pkl_name) 
    Peru_data_temp=readInPredictandLatLon(data_dic,data_dic[var],lat1=-7,lat2=-3,lon1=283,lon2=288,name='Peru',var=var)    
    
    
   # datat=mergeDatasets(site_data,tele_data) #makes sure dates match too
   # datanocluster=mergeDatasets(site_data,NoCluster_data)
    return 1    
    
if __name__ == '__main__':
    main()
 