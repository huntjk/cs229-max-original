 # -*- coding: utf-8 -*-
"""

@author: max

tools to save a dictionary as a .pkl and then to open a dictionary in a .pkl

probably also works for non dictionary objects
"""
import os
import pickle
import cartopy.crs as ccrs
import cartopy.io.shapereader as shapereader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

def savedict(dic, path=''):
    """
    pickle a dictionary.
    """
    ## Get path if not specified in function call
    if not path:
        path = os.curdir
    ## Add correct file extension to path passed with function call
    else:
        if path[-4:] != '.pkl':
            path += '.pkl'
    try:
        ## Open file and dump dictionary        
        with open(path, 'wb') as output:
            pickle.dump(dic, output)
            output.close()    
        return path
    except FileNotFoundError:
        raise

def opendict(path=''):
    """
    unpickle a dictionary.
    """
    ## Get path if not specified in function call
    if not path or not os.path.isfile(path):
        path = os.curdir
        try:
            os.chdir(os.path.split(path)[0])
        ## Exit if invalid path is given or cancel button is pressed
        except OSError:
            raise
    ## open file and load dictionary
    with open(path, 'rb') as picklein:
        dic = pickle.load(picklein)
        picklein.close()    
    return dic 



#visualization code modified from https://matplotlib.org/examples/mplot3d/surface3d_demo3.html
def plot3d(data_dic,skt):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    # Make data.
    Y = data_dic['lat']
    X = data_dic['lon']
    X, Y = np.meshgrid(X, Y)
    Z = skt.T
    
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
    skt=skt.T
    # Make data.
    X = data_dic['lat']
    Y = data_dic['lon']
    X, Y = np.meshgrid(Y, X)
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