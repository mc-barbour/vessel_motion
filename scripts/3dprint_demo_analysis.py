#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 14:46:36 2022

@author: mbarbour

Script to process the 3D printed grafts tested on 05/04/2022 with Andrew

We tested two different grafts with different wall thicknesses: 0.5mm and 1mm

This was a POC test, mostly to see if the printed grafts held water and did not ruputre under pressure

"""


import glob

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
names = ['3Dprint_demo_1mm', '3Dprint_demo_05mm']
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
dt = 1./ experiments[0].frequency
time = np.arange(experiments[0].get_n_images())*dt
datas = [diameter1, diameter2]
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






        