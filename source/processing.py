#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 11:28:47 2022

@author: mbarbour
"""

import numpy as np
import pandas as pd
from skimage import filters, io
import scipy.signal


import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots



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
    
    
    for count,image_file in enumerate(experiment.image_files):
        print("Analyzing Image {:d}".format(count))
        
        image = io.imread(image_file)
        diameter_stations = []
        for station in stations:
            distance, top_index, bottom_index = measure_distance_sobel_vertical(image, station, midpoint)
            diameter_stations.append(distance)

        diameters.append(diameter_stations)

        
    return np.array(diameters)




def analyze_image_stack_windows(experiment, n_stations=10):
    """Process the image stack within a subset of windows"""

    windows = experiment.get_windows()
    stations = np.array([],dtype=int)
    for window in windows:
        window_range = window[1] - window[0]
        if window_range < 0:
            raise Exception("window range is negative, check sheet")
        elif window_range == 720:
            stations = np.linspace(20,experiment.get_image_size()[0]-20, n_stations, dtype=int)
            break
        
        n_stations = int(window_range / 70) + 1
        window_stations = np.linspace(window[0]+5, window[1]-5, n_stations, dtype=int)
        stations = np.append(stations, window_stations)
       
        
    midpoint = int(experiment.get_image_size()[1]/2)

    diameters = []
    for count in range(experiment.start_image, experiment.last_image):
        print("Analyzing Image {:d}".format(count))
        
        image = io.imread(experiment.image_files[count])
        diameter_stations = []
        for station in stations:
            distance, top_index, bottom_index = measure_distance_sobel_vertical(image, station, midpoint)
            diameter_stations.append(distance)

        diameters.append(diameter_stations)
        

    return np.array(diameters)



def waveform_amplitude(data, minimums, maximums):
    """ Change in diameter : Ds - Dd """
    amplitudes = []
    n_measurements = len(minimums)-1
    for count in range(n_measurements):
        min_val = data[minimums[count]]

        if minimums[0] < maximums[0]:
            max_val = data[maximums[count]]
        else:
            max_val = data[maximums[count+1]]

            
        amplitudes.append(max_val - min_val)

    return amplitudes

def waveform_compliance(data, minimums, maximums):
    """ Change in diameter is presented as a percentage"""
    
    amplitudes = []
    n_measurements = len(minimums)-1
    for count in range(n_measurements):
        min_val = data[minimums[count]]

        if minimums[0] < maximums[0]:
            max_val = data[maximums[count]]
        else:
            max_val = data[maximums[count+1]]

            
        amplitudes.append((max_val - min_val) / min_val)

    return amplitudes

def waveform_amplitude_2(data, minimums, maximums):
    amplitudes = []
    
    if len(maximums) == len(minimums):
        points = maximums + minimums
        points.sort()
    elif len(maximums) > len(minimums):
        points = maximums[0:-1] + minimums
        points.sort()
    else:
        points = maximums + minimums[0:-1]
        points.sort()
        
    n_measurements = len(points)
    


def compute_compliance(exp, search_window=100, show_plot=True):
    
    """
    Returns the ratio of diameter motion over initital diameter:
        
        (Ds - Dd)/Dd : s and d are systole and diastole
    
    Computes this value at all analysis stations/windows
    
    """
    
    
    exp.load_pressure()
    pressure = exp.pressure_data[exp.start_image:exp.last_image].values
    df_diameter = exp.load_diameter()
    
    peak_vals_pressure = scipy.signal.find_peaks(pressure, distance=search_window)[0][1:-1]
    min_vals_pressure = scipy.signal.find_peaks(pressure*-1, distance=search_window)[0][1:-1]
    
    # peak_vals_pressure = peak_vals_pressure[1:-1] # skip first and last - rising and falling
    # min_vals_pressure = min_vals_pressure[1:-1] # skip first and last - rising and falling

    print(peak_vals_pressure, min_vals_pressure)
    
    pressure_amplitudes = waveform_amplitude(pressure, min_vals_pressure, peak_vals_pressure)
    
    df_amplitudes = pd.DataFrame(data=pressure_amplitudes, columns=['Pressure (mmHg'])
    
    n_stations = ['station' in a for a in df_diameter.columns.values].count(True)
    peak_vals_all = []
    min_vals_all = []
    for count in range(n_stations):
        
        peak_vals = scipy.signal.find_peaks(df_diameter['station '+str(count+1)], distance=search_window)[0][1:-1]
        min_vals = scipy.signal.find_peaks(df_diameter['station '+str(count+1)]*-1, distance=search_window)[0][1:-1]
        
        
        print(min_vals,peak_vals)
        
        diameter_amplitude = waveform_compliance(df_diameter['station '+str(count+1)],min_vals, peak_vals)
        # diameter_amplitudes_scaled = [a * exp.scale for a in diameter_amplitude]
        
        # df_amplitudes["Diameter, Station "+str(count)] = diameter_amplitudes_scaled       
        df_temp = pd.DataFrame(data = diameter_amplitude, columns = ["Diameter, Station "+str(count+1)])    
        df_amplitudes = pd.concat([df_amplitudes,df_temp],axis=1)

        peak_vals_all.append(peak_vals)
        min_vals_all.append(min_vals)
    
    if show_plot:
        dt = 1./ exp.frequency
        time = np.arange(len(df_diameter))*dt
        fig = make_subplots(rows=2, cols=1)
        for n in range(n_stations):
            fig.add_trace(go.Scatter(x=time,y=df_diameter['station '+str(n+1)], name='station ' + str(n+1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=time[peak_vals_all[n]],y=df_diameter['station '+str(n+1)][peak_vals_all[n]],mode='markers', name='station ' + str(n+1), marker_color='red'), row=1, col=1)
            fig.add_trace(go.Scatter(x=time[min_vals_all[n]],y=df_diameter['station '+str(n+1)][min_vals_all[n]],mode='markers', name='station ' + str(n+1), marker_color='blue'), row=1, col=1)
            fig.update_yaxes(title='Diameter (pixels)', row=1, col=1)


        fig.add_trace(go.Scatter(x=time, y=pressure), row=2, col=1)
        fig.add_trace(go.Scatter(x=time[peak_vals_pressure],y=pressure[peak_vals_pressure], mode='markers'),row=2,col=1)
        fig.add_trace(go.Scatter(x=time[min_vals_pressure],y=pressure[min_vals_pressure], mode='markers', marker_color='green'),row=2,col=1)
        fig.update_yaxes(title='Pressure (mmHg)', row=2, col=1)
        fig.update_xaxes(title='time (s)')
        fig.update_layout(title=exp.get_name())

        fig.update_xaxes(range=[0,time[-1]])
        fig.show()
        
    return df_amplitudes
    


def compute_compliance_stations(exp, stations, search_window=100,show_plot=True):
        
    """
    Returns the ratio of diameter motion over initital diameter:
        
        (Ds - Dd)/Dd : s and d are systole and diastole
    
    Computes this value at specified stations
    
    """
    
    exp.load_pressure()
    pressure = exp.pressure_data[exp.start_image:exp.last_image].values
    df_diameter = exp.load_diameter()
    
    peak_vals_pressure = scipy.signal.find_peaks(pressure, distance=search_window)[0][1:-1]
    min_vals_pressure = scipy.signal.find_peaks(pressure*-1, distance=search_window)[0][1:-1]
    
    # peak_vals_pressure = peak_vals_pressure[1:-1] # skip first and last - rising and falling
    # min_vals_pressure = min_vals_pressure[1:-1] # skip first and last - rising and falling

    print(peak_vals_pressure, min_vals_pressure)
    
    pressure_amplitudes = waveform_amplitude(pressure, min_vals_pressure, peak_vals_pressure)
    
    df_amplitudes = pd.DataFrame(data=pressure_amplitudes, columns=['Pressure (mmHg'])
    
    n_stations = ['station' in a for a in df_diameter.columns.values].count(True)
    peak_vals_all = []
    min_vals_all = []
    
    
    dt = 1./ exp.frequency
    time = np.arange(len(df_diameter))*dt
    fig = make_subplots(rows=2, cols=1)
    
    
    for station in stations:
        
        peak_vals = scipy.signal.find_peaks(df_diameter['station '+str(station)], distance=search_window)[0][1:-1]
        min_vals = scipy.signal.find_peaks(df_diameter['station '+str(station)]*-1, distance=search_window)[0][1:-1]
        
        fig.add_trace(go.Scatter(x=time,y=df_diameter['station '+str(station)], name='station ' + str(station)), row=1, col=1)
        fig.add_trace(go.Scatter(x=time[peak_vals], y=df_diameter['station '+str(station)][peak_vals],mode='markers', name='station ' + str(station), marker_color='red'), row=1, col=1)
        fig.add_trace(go.Scatter(x=time[min_vals], y=df_diameter['station '+str(station)][min_vals],mode='markers', name='station ' + str(station), marker_color='blue'), row=1, col=1)
        fig.update_yaxes(title='Diameter (pixels)', row=1, col=1)
        
        # peak_vals = peak_vals[1:-1]
        # min_vals = min_vals[1:-1]
        
        print(min_vals,peak_vals)
        
        diameter_amplitude = waveform_compliance(df_diameter['station '+str(station)],min_vals, peak_vals)
        # diameter_amplitudes_scaled = [a * exp.scale for a in diameter_amplitude]
        
        # df_amplitudes["Diameter, Station "+str(count)] = diameter_amplitudes_scaled       
        df_temp = pd.DataFrame(data = diameter_amplitude, columns = ["Diameter, Station "+str(station)])    
        df_amplitudes = pd.concat([df_amplitudes, df_temp],axis=1)

        peak_vals_all.append(peak_vals)
        min_vals_all.append(min_vals)
    
    fig.add_trace(go.Scatter(x=time, y=pressure), row=2, col=1)
    fig.add_trace(go.Scatter(x=time[peak_vals_pressure],y=pressure[peak_vals_pressure], mode='markers'),row=2,col=1)
    fig.add_trace(go.Scatter(x=time[min_vals_pressure],y=pressure[min_vals_pressure], mode='markers', marker_color='green'),row=2,col=1)
    fig.update_yaxes(title='Pressure (mmHg)', row=2, col=1)
    fig.update_xaxes(title='time (s)')
    fig.update_layout(title=exp.get_name())

    fig.update_xaxes(range=[0,time[-1]])
    fig.show()
    

        
    return df_amplitudes










