#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 09:56:57 2022

@author: mbarbour
"""


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




#%% Load the two tests
names = ['GORE_6mmID_50_90_v1', 'GORE_6mmID_50_90_v2','GORE_6mmID_80_120_v1', 'GORE_6mmID_80_120_v2','GORE_6mmID_110_150_v1', 'GORE_6mmID_110_150_v2']
experiments = []

for name in names:
    experiments.append(get_experiment(name))
    
#%% compute the diameter - should pass this back to the class and update it

diameter1 = analyze_image_stack(experiments[0])
diameter2 = analyze_image_stack(experiments[1])

#%% load pressure data
for exp in experiments:
    exp.read_pressure()
    
#%% plot the data - exp
stations = np.linspace(20,experiments[0].get_image_size()[0]-20, 10, dtype=int)
dt = 1./ experiments[0].frequency
time = np.arange(experiments[0].get_n_images())*dt
datas = [diameter1, diameter2]
for count,data in enumerate(datas):

    fig = make_subplots(rows=2, cols=1)
    for n in range(len(stations)):
        fig.add_trace(go.Scatter(x=time,y=data[:,n], name='station ' + str(stations[n])), row=1, col=1)
    fig.update_yaxes(title='Diameter (pixels)', row=1, col=1)

    
    
    
    # fig.add_trace(go.Scatter(x=time,y=experiments[count].pressure_data[0:experiments[count].get_n_images()]), row=2, col=1)
    fig.update_yaxes(title='Pressure (mmHg)', row=2, col=1)
    fig.update_layout(title=experiments[count].get_name())
    fig.update_xaxes(title='time (s)')
    fig.show()



#%% Load the two tests
names = ['GORE_6mmID_80_120_v1', 'GORE_6mmID_80_120_v2','GORE_6mmID_110_150_v1', 'GORE_6mmID_110_150_v2']
names = ['GORE_6mmID_50_150_v1', 'GORE_6mmID_50_150_v2', ]
names = ['GORE2_6mmID_50_150_v1', 'GORE2_6mmID_50_150_v2', ]


for name in names:
    experiments.append(get_experiment(name))
    
#%% compute the diameter - should pass this back to the class and update it


# diameter3 = analyze_image_stack(experiments[2])
diameter4 = analyze_image_stack(experiments[3])
diameter5 = analyze_image_stack(experiments[4])
diameter6 = analyze_image_stack(experiments[5])
diameter7 = analyze_image_stack(experiments[6])
diameter8 = analyze_image_stack(experiments[7])
diameter9 = analyze_image_stack(experiments[8])
diameter10 = analyze_image_stack(experiments[9])

#%% load pressure data
for exp in experiments:
    exp.read_pressure()
    
#%% plot the data - exp
stations = np.linspace(20,experiments[0].get_image_size()[0]-20, 10, dtype=int)
dt = 1./ experiments[0].frequency
time = np.arange(experiments[0].get_n_images())*dt
datas = [diameter1, diameter2, diameter3, diameter4, diameter5, diameter6, diameter7, diameter8, diameter9, diameter10]
for count,data in enumerate(datas):

    fig = make_subplots(rows=2, cols=1)
    for n in range(len(stations)):
        fig.add_trace(go.Scatter(x=time,y=data[:,n], name='station ' + str(stations[n])), row=1, col=1)
    fig.update_yaxes(title='Diameter (pixels)', row=1, col=1)
    
    fig.add_trace(go.Scatter(x=time,y=experiments[count].pressure_data[0:experiments[count].get_n_images()]), row=2, col=1)
    fig.update_yaxes(title='Pressure (mmHg)', row=2, col=1)
    fig.update_layout(title=experiments[count].get_name())
    fig.update_xaxes(title='time (s)')
    fig.show()









