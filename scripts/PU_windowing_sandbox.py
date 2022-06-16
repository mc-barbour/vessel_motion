# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots

from skimage import filters, io, feature

from source.experiment import *
from source.processing import *


pio.renderers.default = "browser"

#%%

name = 'PU_02_50_90_RT_v2'
exp = get_experiment(name)
#%% figure out how to read in the processing windows

experiment_log = pd.read_csv(data_dir + 'experiment_log.csv', index_col='name')
params = experiment_log.loc[name]


def get_windows(params):
    window_strs = params['dry windows'].split(',')
    windows = []
    for window_range in window_strs:
        start_ind = int(window_range.split(":")[0][1::])
        stop_ind = int(window_range.split(":")[1][0:-1])
        windows.append([start_ind,stop_ind])
    return windows
        
print(get_windows(params))
#%% now utilize the processing windows in the analysis script

windows = exp.get_windows()
stations = np.array([],dtype=int)
for window in windows:
    window_range = window[1] - window[0]
    if window_range < 0:
        raise Exception("window range is negative, check sheet")
    elif window_range == 720:
        stations = np.linspace(20,experiment.get_image_size()[0]-20, 10, dtype=int)
        break
    
    n_stations = int(window_range / 70) + 1
    window_stations = np.linspace(window[0]+5, window[1]-5, n_stations, dtype=int)
    stations = np.append(stations,window_stations)
   
#%% 
exp.save_diameters(diameter)
