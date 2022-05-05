#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 11:28:47 2022

@author: mbarbour
"""

import numpy as np
from skimage import filters, io



def measure_distance_sobel_horizontal(image, station, midpoint):
    """ compute distance between two edges in an image. graft is horizontal. filter is sobel"""
    sobel = filters.sobel(image)

    top_index = np.argmax(sobel[0:midpoint,station-5:station+5],axis=0)
    
    bottom_index = np.argmax(sobel[midpoint:-1,station-5:station+5],axis=0) + midpoint

    return abs(np.mean(top_index) - np.mean(bottom_index)), np.mean(top_index), np.mean(bottom_index)


def measure_distance_sobel_vertical(image, station, midpoint):
    """ compute distance between two edges in an image. graft is vertical. filter is sobel"""
    
    sobel = filters.sobel(image)
    
    top_index = np.argmax(sobel[station-5:station+5, 0:midpoint],axis=1)

    bottom_index = np.argmax(sobel[station-5:station+5,midpoint:-1],axis=1) + midpoint


    return abs(np.mean(top_index) - np.mean(bottom_index)), np.mean(top_index), np.mean(bottom_index)


def analyze_image_stack(experiment, n_stations=10):
    """Make this into a class: measurement"""
    
    stations = np.linspace(20,experiment.get_image_size()[0]-20, 10, dtype=int)
    midpoint = int(experiment.get_image_size()[1]/2)

    diameters = []
    
    
    for image_file in experiment.image_files:
        
        image = io.imread(image_file)
        diameter_stations = []
        for station in stations:
            distance, top_index, bottom_index = measure_distance_sobel_vertical(image, station, midpoint)
            diameter_stations.append(distance)

        diameters.append(diameter_stations)

        
    return np.array(diameters)