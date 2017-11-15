 # -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 14:30:31 2013

@author: blunghino

tools to save a dictionary as a .pkl and then to open a dictionary in a .pkl

probably also works for non dictionary objects
"""
import os
import pickle

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