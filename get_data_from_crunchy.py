### Import useful routines
import numpy as np
import string
import glob
import os
from collections import Counter
import scipy.stats as stats 
### Import CDAT routines ###
import MV2 as MV
import cdms2 as cdms
import genutil
import cdutil
import cdtime
from eofs.cdms import Eof

### Import scipy routines for smoothing, interpolation
from scipy.interpolate import interp1d
from scipy.optimize import brentq,fminbound
import scipy.ndimage as ndimag

import CMIP5_tools as cmip5
import DA_tools as da
from Plotting import *


### Import plotting routines
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap  
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
#from matplotlib import mpl

### Set classic Netcdf (ver 3)
cdms.setNetcdfShuffleFlag(0)
cdms.setNetcdfDeflateFlag(0)
cdms.setNetcdfDeflateLevelFlag(0)


def sahel_average_hist(X):
    if X.id=="pr":
        X=X*60*60*24
    sahel = cdutil.region.domain(latitude=(10,20),longitude=(-20,30))
    sahel_average=cdutil.averager(X(sahel),axis='xy')(time=('1860-1-1','2006-1-1'))
    return sahel_average
    

def sahel_west_hist(X):
    if X.id=="pr":
        X=X*60*60*24
    sahel_west = cdutil.region.domain(latitude=(10,20),longitude=(-20,0))
    sahel_average_west=cdutil.averager(X(sahel_west),axis='xy')(time=('1860-1-1','2006-1-1'))
    return sahel_average_west

def sahel_east_hist(X):
    if X.id=="pr":
        X=X*60*60*24
    sahel_east = cdutil.region.domain(latitude=(10,20),longitude=(0,30))
    sahel_average_east=cdutil.averager(X(sahel_east),axis='xy')(time=('1860-1-1','2006-1-1'))
    return sahel_average_east


def sahel_average_rcp(X):
    if X.id=="pr":
        X=X*60*60*24
    sahel = cdutil.region.domain(latitude=(10,20),longitude=(-20,30))
    sahel_average=cdutil.averager(X(sahel),axis='xy')(time=('2006-1-1','2100-1-1'))
    return sahel_average
    

def sahel_west_rcp(X):
    if X.id=="pr":
        X=X*60*60*24
    sahel_west = cdutil.region.domain(latitude=(10,20),longitude=(-20,0))
    sahel_average_west=cdutil.averager(X(sahel_west),axis='xy')(time=('2006-1-1','2100-1-1'))
    return sahel_average_west

def sahel_east_rcp(X):
    if X.id=="pr":
        X=X*60*60*24
    sahel_east = cdutil.region.domain(latitude=(10,20),longitude=(0,30))
    sahel_average_east=cdutil.averager(X(sahel_east),axis='xy')(time=('2006-1-1','2100-1-1'))
    return sahel_average_east

def sahel_average_piC(X):
    if X.id=="pr":
        X=X*60*60*24
    sahel = cdutil.region.domain(latitude=(10,20),longitude=(-20,30))
    sahel_average=cdutil.averager(X(sahel),axis='xy')[:200*12]
    return sahel_average
    

def sahel_west_1pct(X):
    if X.id=="pr":
        X=X*60*60*24
    sahel_west = cdutil.region.domain(latitude=(10,20),longitude=(-20,0))
    sahel_average_west=cdutil.averager(X(sahel_west),axis='xy')[:140*12]
    return sahel_average_west

def sahel_east_1pct(X):
    if X.id=="pr":
        X=X*60*60*24
    sahel_east = cdutil.region.domain(latitude=(10,20),longitude=(0,30))
    sahel_average_east=cdutil.averager(X(sahel_east),axis='xy')[:140*12]
    return sahel_average_east
def sahel_average_1pct(X):
    if X.id=="pr":
        X=X*60*60*24
    sahel = cdutil.region.domain(latitude=(10,20),longitude=(-20,30))
    sahel_average=cdutil.averager(X(sahel),axis='xy')[:140*12]
    return sahel_average
    

def sahel_west_piC(X):
    if X.id=="pr":
        X=X*60*60*24
    sahel_west = cdutil.region.domain(latitude=(10,20),longitude=(-20,0))
    sahel_average_west=cdutil.averager(X(sahel_west),axis='xy')[:200*12]
    return sahel_average_west

def sahel_east_piC(X):
    if X.id=="pr":
        X=X*60*60*24
    sahel_east = cdutil.region.domain(latitude=(10,20),longitude=(0,30))
    sahel_average_east=cdutil.averager(X(sahel_east),axis='xy')[:200*12]
    return sahel_average_east
def sahel_indicators(experiment):
    if experiment == "piControl":
        east = cmip5.get_ensemble(experiment,"pr",func=sahel_east_piC)
        west = cmip5.get_ensemble(experiment,"pr",func=sahel_west_piC)
        sav = cmip5.get_ensemble(experiment,"pr",func=sahel_average_piC)
    elif experiment.find("rcp")>=0:
        east = cmip5.get_ensemble(experiment,"pr",func=sahel_east_rcp)
        west = cmip5.get_ensemble(experiment,"pr",func=sahel_west_rcp)
        sav = cmip5.get_ensemble(experiment,"pr",func=sahel_average_rcp)
    elif experiment.find("1pct")>=0:
        east = cmip5.get_ensemble(experiment,"pr",func=sahel_east_1pct)
        west = cmip5.get_ensemble(experiment,"pr",func=sahel_west_1pct)
        sav = cmip5.get_ensemble(experiment,"pr",func=sahel_average_1pct)
    else:
        east = cmip5.get_ensemble(experiment,"pr",func=sahel_east_hist)
        west = cmip5.get_ensemble(experiment,"pr",func=sahel_west_hist)
        sav = cmip5.get_ensemble(experiment,"pr",func=sahel_average_hist)
    
    east.id="pr_CE"
    west.id="pr_W"
    sav.id="pr_sahel"
    f=cdms.open("/kate/SAHEL/cmip5.sahel_precip."+experiment+".nc","w")
    f.write(east)
    f.write(west)
    f.write(sav)
    f.close()

if __name__=="__main__":
    experiments=["piControl","historical","historicalGHG","AA","rcp85"]
    for experiment in experiments:
        sahel_indicators(experiment)
