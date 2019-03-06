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
#from mpl_toolkits.basemap import Basemap  
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
#from matplotlib import mpl

### Set classic Netcdf (ver 3)
cdms.setNetcdfShuffleFlag(0)
cdms.setNetcdfDeflateFlag(0)
cdms.setNetcdfDeflateLevelFlag(0)

from eofs.multivariate.cdms import MultivariateEof
class Fingerprints():
    def __init__(self, experiment):
        self.experiment=experiment
        f = cdms.open("DATA/cmip5.sahel_precip."+experiment+".nc")
        west = f("pr_W")
        east = f("pr_CE")
               
        self.data={}
        self.data["west"]=west
        self.data["east"]=east

        

        

        west_a = cdutil.ANNUALCYCLE.departures(west)
        east_a = cdutil.ANNUALCYCLE.departures(east)


        if experiment != "piControl":
            nmod,nmon=west_a.shape
            west_rs=west_a.reshape((nmod,nmon/12,12))
            west_rs.setAxis(0,west.getAxis(0))
            tax = cdms.createAxis(west.getTime()[6::12])
            tax.id='time'
            tax.designateTime()
            tax.units=west.getTime().units
            west_rs.setAxis(1,tax)

            westsolver = Eof(MV.average(west_rs,axis=0))
        else:
            west_a = da.concatenate_this(west_a,compressed=True)
            nmon, = west_a.shape
            tax=cdms.createAxis(np.arange(west_a.shape[0]))
            tax.units='months since 0001-1-15'
            tax.id='time'
            tax.designateTime()
            taxC=tax.asComponentTime()
            test = [x.torel('days since 0001-1-15').value for x in taxC]
            tax_days = cdms.createAxis(test)
            tax_days.designateTime()
            tax_days.id='time'
            tax_days.units='days since 0001-1-15'
            west_a.setAxis(0,tax_days)
            taxmonthly = cdms.createAxis(west_a.getTime()[6::12])
            taxmonthly.units=west_a.getTime().units
            taxmonthly.designateTime()
            taxmonthly.id='time'
            west_rs=west_a.reshape((nmon/12,12))
            west_rs.setAxis(0,taxmonthly)
            westsolver = Eof(west_rs)

        if experiment != "piControl":
            nmod,nmon=east_a.shape
            east_rs=east_a.reshape((nmod,nmon/12,12))
            east_rs.setAxis(0,east.getAxis(0))
            east_rs.setAxis(1,tax)

            eastsolver = Eof(MV.average(east_rs,axis=0))
        else:
            east_a = da.concatenate_this(east_a,compressed=True)
            east_a.setAxis(0,tax_days)
            nmon, = east_a.shape
            east_rs=east_a.reshape((nmon/12,12))
            east_rs.setAxis(0,taxmonthly)
            eastsolver = Eof(east_rs)


        facwest=da.get_orientation(westsolver)
        faceast = da.get_orientation(eastsolver)

        self.solvers = {}
        self.solvers["east"]=eastsolver
        self.solvers["west"]=westsolver

        self.reshaped = {}
        self.reshaped["east"]=east_rs
        self.reshaped["west"]=west_rs
        
        if len(self.reshaped["west"].shape)>2:
            
            data=[MV.average(cmip5.ensemble2multimodel(self.reshaped["west"]),axis=0),MV.average(cmip5.ensemble2multimodel(self.reshaped["east"]),axis=0) ]
        else:
            data=[self.reshaped["west"],self.reshaped["east"] ]
        msolver=MultivariateEof(data)
        self.solvers["multi"]=msolver
        self.anomalies={}
        self.anomalies["east"]=east_a
        self.anomalies["west"]=west_a


        # ac_east=cdutil.ANNUALCYCLE.climatology(self.data["east"])
        # EAST=self.reshaped["east"]+ac_east.asma()[:,np.newaxis,:]
        # EAST_mma=MV.average(cmip5.ensemble2multimodel(EAST),axis=0)
        # EAST_ensemble = cmip5.ensemble2multimodel(EAST)

        # ac_west=cdutil.ANNUALCYCLE.climatology(self.data["west"])
        # WEST=self.reshaped["west"]+ac_west.asma()[:,np.newaxis,:]
        # WEST_mma=MV.average(cmip5.ensemble2multimodel(WEST),axis=0)
        # WEST_ensemble = cmip5.ensemble2multimodel(WEST)

        # self.ensembles={}
        # self.ensembles["west"]=WEST_ensemble
        # self.ensembles["east"]=EAST_ensemble
        
        

        

def splice(hist,rcp):
    hmodels = cmip5.models(hist)
    rcpmodels = cmip5.models(rcp)
    hrips = [x.split(".")[1]+"."+x.split(".")[3] for x in hmodels]
    rcprips = [x.split(".")[1]+"."+x.split(".")[3] for x in rcpmodels]
    goodrips=np.intersect1d(hrips,rcprips)
    i=0
    labels = []
    nmod=len(goodrips)
    nt = hist.shape[1]+rcp.shape[1]
    H85 = MV.zeros((nmod,nt)+hist.shape[2:])
    for rip in goodrips:
        smoosh=MV.concatenate((hist[hrips.index(rip)],rcp[rcprips.index(rip)]))
        H85[i]=smoosh
        labels += [hmodels[hrips.index(rip)]+" SPLICED WITH "+rcpmodels[rcprips.index(rip)]]
        if i == 0:
            tax=smoosh.getTime()
        i+=1
    modax = cmip5.make_model_axis(labels)
    H85.setAxisList([modax,tax]+hist.getAxisList()[2:])
    return H85


def create_spliced_file():
    fh = cdms.open("DATA/cmip5.sahel_precip.historical.nc")
    hwest = fh("pr_W")
    heast = fh("pr_CE")
    htot = fh("pr_sahel")
    fh.close()

    frcp = cdms.open("DATA/cmip5.sahel_precip.rcp85.nc")
    rcpwest = frcp("pr_W")
    rcpeast = frcp("pr_CE")
    rcptot = frcp("pr_sahel")
    frcp.close()

    fsplice=cdms.open("DATA/cmip5.sahel_precip.historical-rcp85.nc","w")
    swest = splice(hwest,rcpwest)
    swest.id="pr_W"
    fsplice.write(swest)
    
    seast = splice(heast,rcpeast)
    seast.id="pr_CE"
    fsplice.write(seast)

    stot=splice(htot,rcptot)
    stot.id="pr_sahel"
    fsplice.write(stot)

    fsplice.close()

def plot_eastwest(X):
    if len(X.reshaped["west"].shape)>2:
        data=[MV.average(cmip5.ensemble2multimodel(X.reshaped["west"]),axis=0),MV.average(cmip5.ensemble2multimodel(X.reshaped["east"]),axis=0) ]
    else:
        data=[X.reshaped["west"],X.reshaped["east"] ]
    solver=MultivariateEof(data)
    weofs,eeofs=solver.eofs()
    westsolver = weofs[0]
    eastsolver = eeofs[0]
    fac=da.get_orientation(solver)
    
    plt.subplot(211)
    months = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]
    plt.plot(fac*westsolver.asma(),label="WEST")
    plt.plot(eastsolver.asma()*fac,label="EAST")
    plt.xticks(np.arange(12),months)
    plt.legend()
    plt.subplot(212)
    time_plot(fac*solver.pcs()[:,0],label="WEST")


def by_month(X):
    time_i=X.getAxisIds().index('time')
    nm=X.shape[time_i]
    nyears = int(nm/12)
    newtime=(nyears,12)
    d={}
    for i in range(len(X.shape)):
        d[i]=X.shape[i]
    d[time_i]=newtime
    #now make the new shape
    newshape=()
    for i in range(len(X.shape)):
        x=d[i]
        if type(x)==type(()):
            newshape+=x
        else:
            newshape+=(x,)
    Xr=MV.array(X.asma().reshape(newshape))
    axlist = range(len(X.shape))
    for i in axlist:
        if i != time_i:
            Xr.setAxis(i,X.getAxis(i)) 
    monthax = cdms.createAxis(np.arange(12)+1)
    monthax.id="months"
    monthax.months=str(["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"])
    Xr.setAxis(time_i+1,monthax)
    
    yearax = cdms.createAxis(X.getTime()[6::12])
    for att in X.getTime().attributes:
        setattr(yearax,att,X.getTime().attributes[att])
    
    yearax.id="time"
    yearax.designateTime()
    Xr.setAxis(time_i,yearax)
    
    return Xr



def load_data(dataset):
    f = cdms.open("DATA/OBS/PROCESSED/"+dataset+".nc")
   
    obs_w=f("pr_W")
    stop_time =  cmip5.stop_time(obs_w)
    if stop_time.month != 12:
        stop_time=cdtime.comptime(stop_time.year-1,12,31)
    start_time =  cmip5.start_time(obs_w)
    if start_time.month != 1:
        start_time=cdtime.comptime(start_time.year+1,1,1)
    obs_w=obs_w(time=(start_time,stop_time))
    obs_w=by_month(obs_w)
    obs_e=f("pr_CE")
    stop_time =  cmip5.stop_time(obs_e)
    if stop_time.month != 12:
        stop_time=cdtime.comptime(stop_time.year-1,12,31)
    start_time =  cmip5.start_time(obs_e)
    if start_time.month != 1:
        start_time=cdtime.comptime(start_time.year+1,1,1)
    obs_e=obs_e(time=(start_time,stop_time))
    obs_e=by_month(obs_e)
    return [obs_w-MV.average(obs_w,axis=0),obs_e-MV.average(obs_e,axis=0)]
    
   
     
    
    
def TOE(OnePct,piControl,H85,start=None):    
    #Calculate the time of emergence:
    data = [H85.reshaped["west"],H85.reshaped["east"]]
    nmod,nyears,nmonths=H85.reshaped["west"].shape
    P=MV.zeros((nmod,nyears))
    msolver = OnePct.solvers["multi"]
    fac=da.get_orientation(msolver)

    for i in range(nmod):
        to_proj=[H85.reshaped["west"][i],H85.reshaped["east"][i]]
        P[i]=msolver.projectField(to_proj)[:,0]*fac
    P.setAxis(0,H85.reshaped["west"].getAxis(0))
    timeax = H85.reshaped["west"].getAxis(1)
    timeax.id="time"
    P.setAxis(1,timeax)

    piCdata = [piControl.reshaped["west"],piControl.reshaped["east"]]
    pc=msolver.projectField(piCdata)[:,0]
    if start is None:
        start=cdtime.comptime(2000,1,1)
    stop=start.add(1,cdtime.Years)
    final_year=cdtime.comptime(2099,12,31)
    tl=final_year.year-start.year+1

    NOISE = MV.zeros(tl)
    SIGNAL = MV.zeros((nmod,tl))
    i=0
    while stop.cmp(final_year)<0:
        modelproj=P(time=(start,stop))
        L = modelproj.shape[1]
        slopes=da.get_slopes(pc,L)
        SIGNAL[:,i] = cmip5.get_linear_trends(modelproj)
        NOISE[i]=np.ma.std(slopes)
        stop = stop.add(1,cdtime.Years)
        i+=1
    return SIGNAL,NOISE


def proj_aerosols(AA,piControl,H85,start=None,stop=None):

    if start is None:
        start = cdtime.comptime(1945,1,1)
    if stop is None:
        stop=cdtime.comptime(1984,12,31)
    data = [H85.reshaped["west"],H85.reshaped["east"]]
    nmod,nyears,nmonths=H85.reshaped["west"].shape
    P=MV.zeros((nmod,nyears))
    msolver = AA.solvers["multi"]
    fac=da.get_orientation(msolver)

    for i in range(nmod):
        to_proj=[H85.reshaped["west"][i],H85.reshaped["east"][i]]
        P[i]=msolver.projectField(to_proj)[:,0]*fac
    P.setAxis(0,H85.reshaped["west"].getAxis(0))
    timeax = H85.reshaped["west"].getAxis(1)
    timeax.id="time"
    P.setAxis(1,timeax)

    piCdata = [piControl.reshaped["west"],piControl.reshaped["east"]]
    pc=msolver.projectField(piCdata)[:,0]
    Pt = P(time=(start,stop))
    nt=len(Pt.getTime())
    hslopes=cmip5.get_linear_trends(Pt)
    pslopes=da.get_slopes(pc,nt)
    
    
def year_exceeds(X):
    below = np.where(X<1.96)[0]
    if len(below)==0:
        return 0
    else:
        return np.max(below)+1
        


def first_last_decade(X,i):
    months = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]
    WEST_ensemble=X.ensembles["west"]
    EAST_ensemble = X.ensembles["east"]
    DIFF_ensemble=WEST_ensemble-EAST_ensemble
    #models=np.unique([x.split("-")[0] for x in cmip5.models(WEST_ensemble)])
    plt.subplot(131)
    first_west = MV.average(WEST_ensemble[:,:10],axis=1)
    last_west = MV.average(WEST_ensemble[:,-10:],axis=1)
    x=np.arange(12)
    if i != "mma":
        plt.plot(x,first_west[i].asma(),color=cm.Reds(.5),label="W (first decade)")
        plt.plot(x,last_west[i].asma(),color=cm.Reds(.9),label="W (last decade)")
    else:
        plt.plot(x,MV.average(first_west,axis=0).asma(),color=cm.Reds(.5),label="W (first decade)")
        plt.plot(x,MV.average(last_west,axis=0).asma(),color=cm.Reds(.9),label="W (last decade)")
    plt.legend()
    plt.title("WEST")
    plt.xticks(np.arange(12),months)

    plt.subplot(132)
    first_east = MV.average(EAST_ensemble[:,:10],axis=1)
    last_east = MV.average(EAST_ensemble[:,-10:],axis=1)
    x=np.arange(12)
    if i!="mma":
        plt.plot(x,first_east[i].asma(),color=cm.Blues(.5),label="CE (first decade)")
        plt.plot(x,last_east[i].asma(),color=cm.Blues(.9),label="CE (last decade)")
    else:
        plt.plot(x,MV.average(first_east,axis=0).asma(),color=cm.Blues(.5),label="CE (first decade)")
        plt.plot(x,MV.average(last_east,axis=0).asma(),color=cm.Blues(.9),label="CE (last decade)")
    plt.title("EAST")
    plt.xticks(np.arange(12),months)
    plt.legend()
    fig = plt.gcf()
    if i!="mma":
        fig.suptitle(cmip5.models(WEST_ensemble)[i])
    else:
        fig.suptitle("MMA")
    plt.subplot(133)
    first_diff = MV.average(DIFF_ensemble[:,:10],axis=1)
    last_diff = MV.average(DIFF_ensemble[:,-10:],axis=1)
    x=np.arange(12)
    if i!="mma":
        plt.plot(x,first_diff[i].asma(),color=cm.Greens(.5),label="W-CE (first decade)")
        plt.plot(x,last_diff[i].asma(),color=cm.Greens(.9),label="W-CE (last decade)")
    else:
        plt.plot(x,MV.average(first_diff,axis=0).asma(),color=cm.Greens(.5),label="W-CE (first decade)")
        plt.plot(x,MV.average(last_diff,axis=0).asma(),color=cm.Greens(.9),label="W-CE (last decade)")
    plt.legend()
    plt.axhline(0,c='k',ls=":")
    plt.title("DIFF")
    plt.xticks(np.arange(12),months)
        

    

    
                                              
                        
