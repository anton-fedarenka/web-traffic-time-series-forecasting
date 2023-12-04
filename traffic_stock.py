import sys

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import math


def func_smape(Ft, At):    
    denom = (np.abs(At) + np.abs(Ft))/2
    diff = np.abs(Ft - At)/denom
    diff[denom == 0] = 0
    
    return np.nanmean(diff)*100


def prediction_plot(series_true, series_fit, series_pred, ax=None, smape_val=None, text_kwargs = {}, **kwargs):   
   
    assert len(series_true) == len(series_fit) + len(series_pred), 'Inappropriate array sizes'
    
    if ax is None:
        fig, ax = plt.subplots()     
    
    ax = series_true.plot(ax = ax, color='0.25', alpha =0.4, style = '-',marker='.', **kwargs) #, title=f'{name}, n_fourier = {n_fourier}')
    ax = series_fit.plot(ax=ax, label = 'Fitted', color='C0')
    ax = series_pred.plot(ax=ax, label = 'Forecast', color='C3')
    ax.legend(loc = 'upper right')
    
    #text_kwargs.setdefault('smape', dict(size=14))
    text_kwargs.setdefault('prop', dict(size=14))
    text_kwargs.setdefault('frameon', True)    
    text_kwargs.setdefault('loc', 'lower center')
    
    if smape_val:
        at = AnchoredText(
            f"smape = {smape_val:.1f}",
            **text_kwargs
            #prop=dict(size=14),
            #frameon=True,
            #loc="upper left",
            #loc = kwargs['loc']
        )
        at.patch.set_boxstyle("square, pad=0.0")
        ax.add_artist(at)
        
    return ax      

    
def normalization(series): 
    return (series - np.mean(series))/np.std(series)


def norm_v2(series): 
    return series/sum(series)


def invboxcox(y,lmbda):
    if lmbda == 0:
        return(np.exp(y))
    else:
        return(np.exp(np.log(lmbda*y+1)/lmbda))
    
def deriv_filter(series, q):       
    
    y = series.copy()    
    y_median = y.rolling(90,min_periods=1,center=False).median()
    
    median = y.diff().median()
    mean = y.diff().mean()
    cut = y.diff().quantile(q)
    
    y_diff_ex = y.diff()[abs(y.diff()) > cut]    
    
    filtered_points = set()
    i = 0
    
    while y_diff_ex.any(): 
        
        i += y_diff_ex.shape[0]
        filtered_points = filtered_points | set(y_diff_ex.index) 
        
        #print(f'shape = {y_diff_ex.shape}, total = {i}')
        #y.loc[y_diff_ex.index] -= y_diff_ex - y_diff_ex.apply(lambda x: np.sign(x)*5*abs(median))        
        
        #y.drop(y_diff_ex.index,inplace=True)
        bad_index = set(y_diff_ex.index.append(y_diff_ex.index - pd.DateOffset(days=1)).sort_values()) & set(y.index)
        y[bad_index] = y_median[bad_index]
        
        #y[y_diff_ex.index] = y_median[y_diff_ex.index]
        y_diff_ex = y.diff() 
        y_diff_ex = y_diff_ex[abs(y_diff_ex) > cut]
        #print(f'y_diff_ex = {y_diff_ex}')
        
        if(i > y.size) or (len(filtered_points) > y.size/2) :
            break
              
        
    y_median[y.index] = y
        
    return y_median.fillna(series) 


def quantile_filter(series, q):
    
    y_filt = series.copy()
    y_filt_roll = y_filt.rolling(90, min_periods=1,center=True).median()
    y_filt[y_filt > y_filt.quantile(q)] = y_filt_roll[y_filt > y_filt.quantile(q)]
    return y_filt.fillna(y_filt_roll)




class TrainSeriesWrapper:
    
    def __init__(self, series):
        self.first_day = series.notna().idxmax()
        self.series = series.loc[self.first_day:]
        self.log = None
        self.norm = None
        self.train_deriv_filt = None
        self.train_quantile_filt = None
        
        self.filt_func = None
        self.filt_train = None
        
    def normalize(self, ts):
        ts_mean = np.mean(ts)
        ts_std = np.std(ts)
        norm_ts = (ts - ts_mean)/ts_std
        
        return norm_ts, ts_mean, ts_std
        
    def get_train_split(self, n_test, log = True, norm=False):
        
        self.log = log
        self.norm = norm
        self.train_part = self.series.iloc[:-n_test]
        self.test_part = self.series.iloc[-n_test:]
        if self.log: 
            self.train_part = np.log1p(self.train_part)        
        if norm: 
            self.train_part, self.train_mean, self.train_std = self.normalize(self.train_part)
        else:
            self.train_mean = 0
            self.train_std = 1  
       
        return self
    
    def apply_deriv_filter(self, filt_func, **filt_kwargs):
        self.filt_func = filt_func
        self.train_filt = filt_func(self.train_part, **filt_kwargs)
        return self


