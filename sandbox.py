#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 15:56:30 2022

@author: mbarbour
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




#%% load first image
data_dir = '/Users/mbarbour/OneDrive - UW/GraftTesting/Data/'
test = 'GORE_8mm_demo'

half_point = 350

diameter = 446
scale = 10 / 446 # mm / pixel

imageFiles = sorted(glob.glob(data_dir + test + '/Test/*.tif'))

image = io.imread(imageFiles[0])
io.imshow(image)
io.show()

#%% apply edge detection filters


edge_roberts = filters.roberts(image)
edge_sobel = filters.sobel(image)

image_seq = np.array([edge_roberts, edge_sobel])

fig = px.imshow(image_seq, facet_col=0)
fig.show()
#%% 

fig = px.imshow(image)
fig.show(renderer='browser')


#%% look at slice across diameter of tube


slices = np.linspace(600,650,10, dtype=int)

fig = go.Figure()
for n in slices:
    fig.add_trace(go.Scatter(y=edge_sobel[0:half_point,n]))

fig.show()


#%% first pass at processing

# at ten stations, measure the diameter. 
# at each station, average the pixel value at ten pixel station (x) - right next


def compute_distance(filtered_image, station, midpoint):
    
    top_index = np.argmax(filtered_image[0:midpoint,station-5:station+5],axis=0)
    
    bottom_index = np.argmax(filtered_image[midpoint:-1,station-5:station+5],axis=0) + midpoint

    return abs(np.mean(top_index) - np.mean(bottom_index)), np.mean(top_index), np.mean(bottom_index)



#%% batch_process images


n_images = 1000

stations = np.linspace(20,len(image[0,:])-20, 10, dtype=int)

distance_images = []
top_images = []
bottom_images = []
for n in range(n_images):
    

    image = io.imread(imageFiles[n])
    edge_sobel = filters.sobel(image)
    
    distance_stations = []
    top_index_stations = []
    bottom_index_stations = []
    for station in stations:
        distance, top, bottom = compute_distance(edge_sobel,station,half_point)
        distance_stations.append(distance)
        top_index_stations.append(top)
        bottom_index_stations.append(bottom)

    bottom_images.append(bottom_index_stations)
    top_images.append(top_index_stations)
    distance_images.append(distance_stations)
    
dist_array = np.array(distance_images)
top_array = np.array(top_images)
bottom_array = np.array(bottom_images)


#%% batch_process images - just a single point in the image thought

half_point=350

n_images = 1000

stations = np.linspace(20,len(image[0,:])-20, 10, dtype=int)

distance = []
top = []
bottom = []
for n in range(n_images):
    

    image = io.imread(imageFiles[n])
    edge_sobel = filters.sobel(image)

    
    distance_val, top_val, bottom_val = compute_distance(edge_sobel,600,half_point)
    distance.append(distance_val)
    top.append(top_val)
    bottom.append(bottom_val)

fig = go.Figure()

fig.add_trace(go.Scatter(y=distance))
fig.update_yaxes(title='Distance ')
fig.show()



fig = go.Figure()
fig.add_trace(go.Scatter(y=top))
fig.update_yaxes(title='Top')
fig.show()




#%% plot

fig = go.Figure()
for n in range(len(stations)):
    fig.add_trace(go.Scatter(y=dist_array[:,n] * scale, name='station ' + str(stations[n])))
fig.update_yaxes(title='Distance (mm)')
fig.show()



fig = go.Figure()
for n in range(len(stations)):
    fig.add_trace(go.Scatter(y=top_array[:,n], name='station ' + str(stations[n])))
fig.update_yaxes(title='Top')
fig.show()



#%% process a single image
image_num = 995
imageFile = glob.glob(data_dir + test + '/Test/Test0'+str(image_num)+'.tif')[0]
image = io.imread(imageFile)
edge_sobel = filters.sobel(image)

fig = px.imshow(image)
fig.show()

fig = px.imshow(edge_sobel)
fig.show()
#%%
slices = np.linspace(600,650,10, dtype=int)
station=625
fig = go.Figure()
for n in slices:
    fig.add_trace(go.Scatter(y=edge_sobel[0:half_point,n]))

fig.show()


print(np.argmax(edge_sobel[:,station]))

print(np.argmax(edge_sobel[0:350,station-5:station+5],axis=0))



distance, top, bottom = compute_distance(edge_sobel, 625, 350)


#%% Load and plot pressure
freq = 100
dt = 1. / freq

time = np.arange(n_images)*dt

df = pd.read_csv(data_dir + test + '/GORE_graft_demo_test2.txt', names=('Index', 'Pressure (mmHg)'))

fig = make_subplots(rows=2, cols=1)

fig.add_trace(go.Scatter(x=time,y=dist_array[:,8] * scale), row=1,col=1)
fig.add_trace(go.Scatter(x=time,y=dist_array[:,9] * scale), row=1,col=1)
fig.add_trace(go.Scatter(x=time,y=dist_array[:,7] * scale), row=1,col=1)

fig.add_trace(go.Scatter(x=time,y=df['Pressure (mmHg)'][0:n_images]), row=2,col=1)


fig.update_yaxes(title='Diameter (mm)', row=1, col=1)
fig.update_yaxes(title='Pressure (mmHg)', row=2, col=1)
fig.update_xaxes(title='Time (s)', row=2, col=1)


fig.show()


#%% test experiment class

exp = get_experiment('3Dprint_demo_1mm')

#%% compute the edges of a sample image
# should try canny image filter
# contour detection

image = io.imread(exp.image_files[0])
sobel = filters.sobel(image)
canny = feature.canny(image)

fig = make_subplots(rows=1, cols=3)
fig.add_trace(px.imshow(image, zmin=0, zmax=200).data[0], 1, 1)
# fig.add_trace(px.imshow(sobel, zmax=0.2).data[0], 1, 2)
fig.add_trace(px.imshow(canny, zmax=10).data[0], 1, 3)

fig.show()


#%%
n_images = 1000

stations = np.linspace(20,len(image[0,:])-20, 10, dtype=int)

distance_images = []
top_images = []
bottom_images = []
for n in range(n_images):
    

    image = io.imread(imageFiles[n])
    edge_sobel = filters.sobel(image)
    
    distance_stations = []

    for station in stations:
        distance, top, bottom = compute_distance(edge_sobel,station,half_point)
        distance_stations.append(distance)
        top_index_stations.append(top)
        bottom_index_stations.append(bottom)

    bottom_images.append(bottom_index_stations)
    top_images.append(top_index_stations)
    distance_images.append(distance_stations)
    
dist_array = np.array(distance_images)
top_array = np.array(top_images)
bottom_array = np.array(bottom_images)

#%% compute distance on a single image - vertical 

station = 425
midpoint = 600

filtered_image = sobel

top_index = np.argmax(filtered_image[station-5:station+5, 0:midpoint],axis=1)

bottom_index = np.argmax(filtered_image[station-5:station+5,midpoint:-1],axis=1) + midpoint

print(abs(np.mean(top_index) - np.mean(bottom_index)), np.mean(top_index), np.mean(bottom_index))
#%% plot

slices = np.linspace(400,450,10, dtype=int)

fig = go.Figure()
for n in slices:
    fig.add_trace(go.Scatter(y=sobel[n,:]))

fig.show()



