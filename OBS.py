
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

import fingerprints as fp


class ObsData():
    def __init__(self,dataset):
        f = cdms.open("DATA/OBS/PROCESSED/"+dataset+".nc")
        self.data={}
        obs_w=f("pr_W")
        self.data["west"]=obs_w
        stop_time =  cmip5.stop_time(obs_w)
        if stop_time.month != 12:
            stop_time=cdtime.comptime(stop_time.year-1,12,31)
        start_time =  cmip5.start_time(obs_w)
        if start_time.month != 1:
            start_time=cdtime.comptime(start_time.year+1,1,1)
        obs_w=obs_w(time=(start_time,stop_time))
        obs_w=fp.by_month(obs_w)
        obs_e=f("pr_CE")
        self.data["east"]=obs_e
        stop_time =  cmip5.stop_time(obs_e)
        if stop_time.month != 12:
            stop_time=cdtime.comptime(stop_time.year-1,12,31)
        start_time =  cmip5.start_time(obs_e)
        if start_time.month != 1:
            start_time=cdtime.comptime(start_time.year+1,1,1)
        obs_e=obs_e(time=(start_time,stop_time))
        obs_e=fp.by_month(obs_e)
        self.reshaped={}
        self.reshaped["east"]=obs_e-MV.average(obs_e,axis=0)
        self.reshaped["west"]=obs_w-MV.average(obs_w,axis=0)
        self.reshaped["multi"]=[self.reshaped["west"],self.reshaped["east"]]
        self.dataset=dataset
        
        
        
        


def get_colors(label):
    d={}
    d["west"]="#1f77b4"
    d["east"]="#ff7f0e"
    d["multi"]="#2ca02c"
    return d[label]

class Sahel():
    def __init__(self,aerosol=None,ghg=None,h85=None,piC=None):
        if aerosol is not None:
            self.aerosol=aerosol
        else:
            self.aerosol=fp.Fingerprints("AA")
        if ghg is not None:
            self.ghg=ghg
        else:
            self.ghg=fp.Fingerprints("1pctCO2")
        if h85 is not None:
            self.h85=h85
        else:
            self.h85 = fp.Fingerprints("historical-rcp85")
        if piC is not None:
            self.piC=piC
        else:
            self.piC = fp.Fingerprints("piControl")  
        
        self.gpcp=ObsData("GPCP")
        self.precl=ObsData("PRECL")
        self.cmap=ObsData("CMAP")
        self.OBS={}
        self.OBS["GPCP"]=self.gpcp
        self.OBS["PRECL"]=self.precl
        self.OBS["CMAP"]=self.cmap
        
        
    def plot_obs_trends(self,dataset,**kwargs):
        X=self.OBS[string.upper(dataset)]
        west=X.reshaped["west"]
        east=X.reshaped["east"]
        if "start" not in kwargs.keys():
            start=cmip5.start_time(east)
            start=cdtime.comptime(start.year,start.month,1)
        else:
            start=kwargs.pop("start")

        if "stop" not in kwargs.keys():
            stop=cmip5.stop_time(east)
            stop=cdtime.comptime(stop.year,stop.month,30)
        else:
            stop=kwargs.pop("stop")
        west=west(time=(start,stop))
        east=east(time=(start,stop))
        west.getAxis(0).id="time"
        east.getAxis(0).id="time"
        plt.plot(cmip5.get_linear_trends(west).asma(),label="WEST",color=get_colors("west"),**kwargs)
        plt.plot(cmip5.get_linear_trends(east).asma(),label="EAST",color=get_colors("east"),**kwargs)
        months = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]
        plt.xticks(np.arange(12),months)

    def plot_model_trends(self,i,start=None,stop=None):
        if i!= "avg":
            west = self.h85.reshaped["west"][i]
            east = self.h85.reshaped["east"][i]
        else:
            west = MV.average( h85.reshaped["west"],axis=0)
            east = MV.average( h85.reshaped["east"],axis=0)
        if start is not None:
            west=west(time=(start,stop))
            east=east(time=(start,stop))
        plt.plot(cmip5.get_linear_trends(west).asma(),label="WEST")
        plt.plot(cmip5.get_linear_trends(east).asma(),label="EAST")
        months = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]
        plt.xticks(np.arange(12),months)
    def project_east_west(self,dataset,experiment,best_fit=True):
        X=self.OBS[string.upper(dataset)]
        fingerprint=getattr(self,experiment)
        westsolver=fingerprint.solvers["west"]
        westfac=da.get_orientation(westsolver)
        time_plot(westfac*westsolver.projectField(X.reshaped["west"])[:,0],label="WEST",color=get_colors("west"))

        eastsolver=fingerprint.solvers["east"]
        eastfac=da.get_orientation(eastsolver)
        time_plot(eastfac*eastsolver.projectField(X.reshaped["east"])[:,0],label="EAST",color=get_colors("east"))

        plt.legend()
        plt.ylabel("Projection onto "+fingerprint.experiment+" fingerprint")
        if best_fit:
            y=westfac*westsolver.projectField(X.reshaped["west"])[:,0]
            t=cmip5.get_plottable_time(y)
            p=np.polyfit(t,y.asma(),1)
            plt.plot(t,np.polyval(p,t),"--",color=get_colors("west"))

            y=eastfac*eastsolver.projectField(X.reshaped["east"])[:,0]
            t=cmip5.get_plottable_time(y)
            p=np.polyfit(t,y.asma(),1)
            plt.plot(t,np.polyval(p,t),"--",color=get_colors("east"))
    def model_projections(self,experiment,direction):
        fingerprint=getattr(self,experiment)
        solver = fingerprint.solvers[direction]
        fac=da.get_orientation(solver)
        if direction == "multi":
            modeldata=[self.h85.reshaped["west"],self.h85.reshaped["east"]]
            shaper=self.h85.reshaped["west"]
        else:
            modeldata =self.h85.reshaped[direction]
            shaper=modeldata
        P=MV.zeros(shaper.shape[:-1])+1.e20
        for i in range(shaper.shape[0]):
            try:
                if direction != "multi":
                    P[i]=fac*solver.projectField(modeldata[i])[:,0]
                else:
                    P[i]=fac*solver.projectField([self.h85.reshaped["west"][i],self.h85.reshaped["east"][i]])[:,0]
            except:
               continue

        Pm=MV.masked_where(np.abs(P)>1.e10,P)
        Pm.setAxisList(shaper.getAxisList()[:-1])
        Pm.getAxis(1).id="time"
        return Pm

    
    def noise_projections(self,experiment,direction):
        fingerprint=getattr(self,experiment)
        if direction == "multi":
            data = [self.piC.reshaped["west"],self.piC.reshaped["east"]]
        else:
            data = self.piC.reshaped[direction]
        solver = fingerprint.solvers[direction]
        fac = da.get_orientation(solver)
        return fac*solver.projectField(data)[:,0]

    def obs_projections(self,experiment,dataset,direction):
        fingerprint=getattr(self,experiment)
        X=self.OBS[string.upper(dataset)]
        solver = fingerprint.solvers[direction]
        fac=da.get_orientation(solver)
        data = X.reshaped[direction]
        return fac*solver.projectField(data)[:,0]
    def DA_histogram(self,experiment,direction,start=None,stop=None):
        fingerprint=getattr(self,experiment)
        
        if start is None:
            start=cmip5.start_time(self.gpcp.reshaped["east"])
            start=cdtime.comptime(start.year,start.month,1)
        if stop is None:
            stop=cmip5.stop_time(self.gpcp.reshaped["east"])
            stop=cdtime.comptime(stop.year,stop.month,30)
        
       
        #get the h85 projections over the same time period
        H85m=self.model_projections(experiment,direction)(time=(start,stop))
        H85=cmip5.cdms_clone(np.ma.mask_rows(H85m),H85m)
        H85_trends=cmip5.get_linear_trends(H85)
        #get the piControl projection time series
        noise=self.noise_projections(experiment,direction)
        L=stop.year-start.year+1
        noise_trends = da.get_slopes(noise,L)

        #plot
        plt.hist(H85_trends.compressed(),25,color=da_colors("h85"),alpha=.5,normed=True)
        plt.hist(noise_trends,25,color=da_colors("piC"),alpha=.5,normed=True)
        da.fit_normals_to_data(H85_trends,color=da_colors("h85"),lw=3,label="H85")
        da.fit_normals_to_data(noise_trends,color=da_colors("piC"),lw=3,label="piControl")
       # plt.axvline(obs_trend,label=obs.dataset,color=da_colors(obs.dataset))

        #Project the observations
        for dataset in ["gpcp","cmap","precl"]:
            
            obs_proj=self.obs_projections(experiment,dataset,direction)(time=(start,stop))
            obs_trend = cmip5.get_linear_trends(obs_proj)
            plt.axvline(obs_trend,label=dataset,color=da_colors(dataset))

    
    def average_histogram(self,direction,start=None,stop=None,months="JJ"):
        if months is "JJ":
            mmean=lambda x: MV.average(x[:,5:7],axis=1)
            bigmmean=lambda X: MV.average(X[:,:,5:7],axis=2)
        elif months is "SO":
            mmean=lambda x: MV.average(x[:,8:10],axis=1)
            bigmmean=lambda X: MV.average(X[:,:,8:10],axis=2)
        elif months is "JJA":
            mmean=lambda x: MV.average(x[:,5:8],axis=1)
            bigmmean=lambda X: MV.average(X[:,:,5:8],axis=2)
        elif months is "Jun":
            mmean=lambda x: x[:,5]
            bigmmean=lambda X: MV.average(X[:,:,5])
        elif months is "YEAR":
            mmean=lambda x: MV.average(x,axis=1)
            bigmmean=lambda X: MV.average(X,axis=2)
        if start is None:
            start=cmip5.start_time(self.gpcp.reshaped["east"])
            start=cdtime.comptime(start.year,start.month,1)
        if stop is None:
            stop=cmip5.stop_time(self.gpcp.reshaped["east"])
            stop=cdtime.comptime(stop.year,stop.month,30)
        
        
        #get the h85 trends over the same time period
        H85m=bigmmean(self.h85.reshaped[direction])(time=(start,stop))
        H85=cmip5.cdms_clone(np.ma.mask_rows(H85m),H85m)
        H85_trends=cmip5.get_linear_trends(H85)
        #get the piControl projection time series
        noise=mmean(self.piC.reshaped[direction])
        L=stop.year-start.year+1
        noise_trends = da.get_slopes(noise,L)

        #plot
        plt.hist(H85_trends.compressed(),25,color=da_colors("h85"),alpha=.5,normed=True)
        plt.hist(noise_trends,25,color=da_colors("piC"),alpha=.5,normed=True)
        da.fit_normals_to_data(H85_trends,color=da_colors("h85"),lw=3,label="H85")
        da.fit_normals_to_data(noise_trends,color=da_colors("piC"),lw=3,label="piControl")
        
        #calculate the trend in the observations
        for dataset in ["gpcp","cmap","precl"]:
            X=self.OBS[string.upper(dataset)]
            obs_avg=mmean(X.reshaped[direction](time=(start,stop)))

            obs_trend = cmip5.get_linear_trends(obs_avg)

            plt.axvline(obs_trend,label=dataset,color=da_colors(dataset))
        plt.xlabel("S/N")
        plt.ylabel("Frequency")
        plt.legend(loc=0)




# def plot_obs_trends(X,**kwargs):
#     west=X.reshaped["west"]
#     east=X.reshaped["east"]
#     if "start" not in kwargs.keys():
#         start=cmip5.start_time(east)
#         start=cdtime.comptime(start.year,start.month,1)
#     else:
#         start=kwargs.pop("start")
    
#     if "stop" not in kwargs.keys():
#         stop=cmip5.stop_time(east)
#         stop=cdtime.comptime(stop.year,stop.month,30)
#     else:
#         stop=kwargs.pop("stop")
#     west=west(time=(start,stop))
#     east=east(time=(start,stop))
#     west.getAxis(0).id="time"
#     east.getAxis(0).id="time"
#     plt.plot(cmip5.get_linear_trends(west).asma(),label="WEST",color=get_colors("west"),**kwargs)
#     plt.plot(cmip5.get_linear_trends(east).asma(),label="EAST",color=get_colors("east"),**kwargs)
#     months = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]
#     plt.xticks(np.arange(12),months)
# def plot_model_trends(h85,i,start=None,stop=None):
#     if i!= "avg":
#         west = h85.reshaped["west"][i]
#         east = h85.reshaped["east"][i]
#     else:
#         west = MV.average( h85.reshaped["west"],axis=0)
#         east = MV.average( h85.reshaped["east"],axis=0)
#     if start is not None:
#         west=west(time=(start,stop))
#         east=east(time=(start,stop))
#     plt.plot(cmip5.get_linear_trends(west).asma(),label="WEST")
#     plt.plot(cmip5.get_linear_trends(east).asma(),label="EAST")
#     months = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]
#     plt.xticks(np.arange(12),months)
        
# def project_east_west(X,fingerprint,best_fit=True):
#     westsolver=fingerprint.solvers["west"]
#     westfac=da.get_orientation(westsolver)
#     time_plot(westfac*westsolver.projectField(X.reshaped["west"])[:,0],label="WEST",color=get_colors("west"))

#     eastsolver=fingerprint.solvers["east"]
#     eastfac=da.get_orientation(eastsolver)
#     time_plot(eastfac*eastsolver.projectField(X.reshaped["east"])[:,0],label="EAST",color=get_colors("east"))
    
#     plt.legend()
#     plt.ylabel("Projection onto "+fingerprint.experiment+" fingerprint")
#     if best_fit:
#         y=westfac*westsolver.projectField(X.reshaped["west"])[:,0]
#         t=cmip5.get_plottable_time(y)
#         p=np.polyfit(t,y.asma(),1)
#         plt.plot(t,np.polyval(p,t),"--",color=get_colors("west"))
        
#         y=eastfac*eastsolver.projectField(X.reshaped["east"])[:,0]
#         t=cmip5.get_plottable_time(y)
#         p=np.polyfit(t,y.asma(),1)
#         plt.plot(t,np.polyval(p,t),"--",color=get_colors("east"))

def project_multi(X,fingerprint,best_fit=True):
    multisolver=fingerprint.solvers["multi"]
    multifac=da.get_orientation(multisolver)
    time_plot(multifac*multisolver.projectField(X)[:,0],label="MULTI",color=get_colors("multi"))

    
    
    plt.legend()
    plt.ylabel("Projection onto "+fingerprint.experiment+" fingerprint")
    if best_fit:
        y=multifac*multisolver.projectField(X)[:,0]
        t=cmip5.get_plottable_time(y)
        p=np.polyfit(t,y.asma(),1)
        plt.plot(t,np.polyval(p,t),"--",color=get_colors("multi"))
        

def model_projections(fingerprint,h85,direction):
    solver = fingerprint.solvers[direction]
    fac=da.get_orientation(solver)
    if direction == "multi":
        modeldata=[h85.reshaped["west"],h85.reshaped["east"]]
        shaper=h85.reshaped["west"]
    else:
        modeldata =h85.reshaped[direction]
        shaper=modeldata
    P=MV.zeros(shaper.shape[:-1])+1.e20
    for i in range(shaper.shape[0]):
        try:
            if direction != "multi":
                P[i]=fac*solver.projectField(modeldata[i])[:,0]
            else:
                P[i]=fac*solver.projectField([h85.reshaped["west"][i],h85.reshaped["east"][i]])[:,0]
        except:
           continue
            
    Pm=MV.masked_where(np.abs(P)>1.e10,P)
    Pm.setAxisList(shaper.getAxisList()[:-1])
    Pm.getAxis(1).id="time"
    return Pm
def plot_model_projections(fingerprint,h85):
    Pwest=model_projections(fingerprint,h85,"west")
    time_plot(MV.average(Pwest,axis=0),color=get_colors("west"),label="WEST")

    Peast=model_projections(fingerprint,h85,"east")
    time_plot(MV.average(Peast,axis=0),color=get_colors("east"),label="EAST")

    Pmulti=model_projections(fingerprint,h85,"multi")
    time_plot(MV.average(Pmulti,axis=0),color=get_colors("multi"),label="MULTI")
    
    
def noise_projections(fingerprint,piC,direction):
    if direction == "multi":
        data = [piC.reshaped["west"],piC.reshaped["east"]]
    else:
        data = piC.reshaped[direction]
    solver = fingerprint.solvers[direction]
    fac = da.get_orientation(solver)
    return fac*solver.projectField(data)[:,0]
    
def obs_projections(fingerprint,X,direction):
    solver = fingerprint.solvers[direction]
    fac=da.get_orientation(solver)
    data = X.reshaped[direction]
    return fac*solver.projectField(data)[:,0]
#detection and attribution
def da_colors(typ):
    d={}
    d["h85"]=cm.Oranges(.8)#cm.Dark2(0.)
    d["piC"]=cm.Greens(.7)#cm.Dark2(.2)
    d["gpcp"]=cm.Purples(.5)#cm.Dark2(.4)
    d["cmap"]=cm.Reds(.8)
    d["precl"]=cm.Purples(.9)
    return d[typ]

def DA_histogram(fingerprint,obslist,h85,piC,direction,start=None,stop=None):
    if type(obslist)==type([]):
        obs=obslist[0]
    else:
        obs=obslist
    if start is None:
        start=cmip5.start_time(obs.reshaped["east"])
        start=cdtime.comptime(start.year,start.month,1)
    if stop is None:
        stop=cmip5.stop_time(obs.reshaped["east"])
        stop=cdtime.comptime(stop.year,stop.month,30)
    #project the observations onto the fingerprint
    obs_proj=obs_projections(fingerprint,obs,direction)(time=(start,stop))
    obs_trend = cmip5.get_linear_trends(obs_proj)
    #get the h85 projections over the same time period
    H85m=model_projections(fingerprint,h85,direction)(time=(start,stop))
    H85=cmip5.cdms_clone(np.ma.mask_rows(H85m),H85m)
    H85_trends=cmip5.get_linear_trends(H85)
    #get the piControl projection time series
    noise=noise_projections(fingerprint,piC,direction)
    L = len(obs_proj)
    noise_trends = da.get_slopes(noise,L)

    #plot
    plt.hist(H85_trends.compressed(),25,color=da_colors("h85"),alpha=.5,normed=True)
    plt.hist(noise_trends,25,color=da_colors("piC"),alpha=.5,normed=True)
    da.fit_normals_to_data(H85_trends,color=da_colors("h85"),lw=3,label="H85")
    da.fit_normals_to_data(noise_trends,color=da_colors("piC"),lw=3,label="piControl")
    plt.axvline(obs_trend,label=obs.dataset,color=da_colors(obs.dataset))
    if type(obslist)==type([]):
        for obs in obslist[1:]:
            
            obs_proj=obs_projections(fingerprint,obs,direction)(time=(start,stop))
            obs_trend = cmip5.get_linear_trends(obs_proj)
            plt.axvline(obs_trend,label=obs.dataset,color=da_colors(obs.dataset))
    return H85,noise,obs_proj
    
def average_histogram(obslist,h85,piC,direction,start=None,stop=None,months="JJ"):
    if months is "JJ":
        mmean=lambda x: MV.average(x[:,5:7],axis=1)
        bigmmean=lambda X: MV.average(X[:,:,5:7],axis=2)
    elif months is "SO":
        mmean=lambda x: MV.average(x[:,8:10],axis=1)
        bigmmean=lambda X: MV.average(X[:,:,8:10],axis=2)
    elif months is "JJA":
        mmean=lambda x: MV.average(x[:,5:8],axis=1)
        bigmmean=lambda X: MV.average(X[:,:,5:8],axis=2)
    elif months is "Jun":
        mmean=lambda x: x[:,5]
        bigmmean=lambda X: MV.average(X[:,:,5])
    if type(obslist)==type([]):
        obs=obslist[0]
    else:
        obs=obslist
    if start is None:
        start=cmip5.start_time(obs.reshaped["east"])
        start=cdtime.comptime(start.year,start.month,1)
    if stop is None:
        stop=cmip5.stop_time(obs.reshaped["west"])
        stop=cdtime.comptime(stop.year,stop.month,30)
    #calculate the trend in the observations
    
   
    obs_avg=mmean(obs.reshaped[direction](time=(start,stop)))
    
    obs_trend = cmip5.get_linear_trends(obs_avg)
    #get the h85 trends over the same time period
    H85m=bigmmean(h85.reshaped[direction])(time=(start,stop))
    H85=cmip5.cdms_clone(np.ma.mask_rows(H85m),H85m)
    H85_trends=cmip5.get_linear_trends(H85)
    #get the piControl projection time series
    noise=mmean(piC.reshaped[direction])
    L = len(obs_avg)
    noise_trends = da.get_slopes(noise,L)

    #plot
    plt.hist(H85_trends.compressed(),25,color=da_colors("h85"),alpha=.5,normed=True)
    plt.hist(noise_trends,25,color=da_colors("piC"),alpha=.5,normed=True)
    da.fit_normals_to_data(H85_trends,color=da_colors("h85"),lw=3,label="H85")
    da.fit_normals_to_data(noise_trends,color=da_colors("piC"),lw=3,label="piControl")
    plt.axvline(obs_trend,label=obs.dataset,color=da_colors(obs.dataset))
    if type(obslist)==type([]):
        for obs in obslist[1:]:
            obs_avg=mmean(obs.reshaped[direction](time=(start,stop)))
    
            obs_trend = cmip5.get_linear_trends(obs_avg)
            
            plt.axvline(obs_trend,label=obs.dataset,color=da_colors(obs.dataset))
    plt.xlabel("S/N")
    plt.ylabel("Frequency")
    plt.legend(loc=0)
            
    
def plot_h85_trends(H85):
    x = cmip5.get_plottable_time(H85)
    nmod=H85.shape[0]
    cmap=cm.magma
    for i in range(nmod):
        if not H85[i].mask[0]:
            time_plot(H85[i],lw=.5,alpha=.5,color=cm.magma(float(i)/float(nmod)))
            p=np.ma.polyfit(x,H85[i],1)
            plt.plot(x,np.polyval(p,x),color=cm.magma(float(i)/float(nmod)))


# TO DO:
#1. Do histograms of observed and modeled trends for total,west, east sahel JAS averages
#2. Are observed changes really much bigger?


#Yearly data
