#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 10:22:24 2022

@author: mbarbour
"""


import pandas as pd
import glob
import os

from skimage import io


from paths import data_dir


# this isnt called on import...
experiment_log = pd.read_csv(data_dir + 'experiment_log.csv', index_col='name')

def get_experiment(name):
    """Return a single experiment class if available in list of experiments"""
    experiment_log = pd.read_csv(data_dir + 'experiment_log.csv', index_col='name')

    if name in experiment_log.index.values:
        return Experiment(name, experiment_log.loc[name])
    else:
        raise ValueError("No experiment found with name {:s}. Check spelling and paths in experiment.py".format(name))
        
def get_experiment_names():
    """"Print all available experiments"""
    experiment_log = pd.read_csv(data_dir + 'experiment_log.csv', index_col='name')

    names = []
    for name in experiment_log.index.values:
        names.append(name)
        print("{:s}".format(name))
    return names


class Experiment():
    """
    Experiment class
    """
    
    def __init__(self, name, params):
        
        self._name = name 
        self._path = data_dir + params['data folder name']
        self.pressure_file = self._path + "/" + params['pressure file']
        self.frequency = params['frequency (Hz)']
        self.image_files = self.get_image_files()
        
        self.set_scale(params['scale (mm/pixel)'])
        self.window_string = params['dry windows']
        self.last_image = int(params['stop image'])
        self.start_image = int(params['start image'])
        self.skip_traces = self.get_skip_traces(params['trace, omit'])
        


    def get_name(self):
        return self._name
    
    def get_path(self):
        return self._path
    
    def get_n_images(self):
        return len(self.image_files)
    
    def get_image_files(self):
        image_dir = self._path + "/Images/"
        image_files = sorted(glob.glob(image_dir + "*.tif"))
        
        if not image_files:
            raise ValueError("No images found in {:s}. Check folder definitions".format(image_dir))
        
        return image_files
    
    def set_scale(self, scale):
        if pd.isna(scale):
            self.scale = 1
            self.units = 'pixels'
            print("Warning, scale is not defined for this experiment, units remain in pixels")
        else:
            self.scale = scale
            self.units = 'mm'
    
    def get_image_size(self):
        image = io.imread(self.image_files[0])
        return image.shape
        

    def load_pressure(self):
        df = pd.read_csv(self.pressure_file, names=('Index', 'Pressure (mmHg)'))
        self.pressure_data = df["Pressure (mmHg)"]
        
    def get_windows(self):
        window_strings = self.window_string.split(",")
        windows = []
        for window_range in window_strings:
            start_ind = int(window_range.split(":")[0][1::])
            stop_ind = int(window_range.split(":")[1][0:-1])
            windows.append([start_ind,stop_ind])
        return windows
    
    def save_diameters(self, diameters, OVERWRITE=False):
        save_name = self._path + "ProcessedDiameters.csv"
        n_stations = len(diameters[0,:])
        columns = ['station '+str(a+1) for a in range(n_stations)]
        df = pd.DataFrame(data=diameters, columns=columns)
        
        if not os.path.exists(save_name):
            
            df.to_csv(save_name)
        elif OVERWRITE:
            print("Saving Diameters File")
            df.to_csv(save_name)
        else:
            print("Not Saving, as file already exists. CHange OVERWRITE to true if you'd like to overwrite")
            
    
    def load_diameter(self):
        filename = self._path + "ProcessedDiameters.csv"
        if not os.path.exists(filename):
            raise Exception("Diameters file does not exist")
        else:
            df = pd.read_csv(filename)
            return df
        
    def save_amplitudes(self, df, OVERWRITE=False):
        save_name = self._path + "Amplitudes.csv"
        
        if not os.path.exists(save_name):
            print("Saving Amplitude  File")
            df.to_csv(save_name)
        elif OVERWRITE:
            print("Saving Amplitude  File")
            df.to_csv(save_name)
        else:
            print("Not Saving, as file already exists. CHange OVERWRITE to true if you'd like to overwrite")
            
    def load_amplitudes(self):
        save_name = self._path + "Amplitudes.csv"
        return pd.read_csv(save_name)
               
            
    def save_compliance(self, df, OVERWRITE=False):
        save_name = self._path + "_Compliance.csv"
        
        if not os.path.exists(save_name):
            print("Saving Compliance  File")
            df.to_csv(save_name)
        elif OVERWRITE:
            print("Saving Compliance  File")
            df.to_csv(save_name)
        else:
            print("Not Saving, as file already exists. CHange OVERWRITE to true if you'd like to overwrite")
    
    def load_compliance(self):
        save_name = self._path + "_Compliance.csv"
        return pd.read_csv(save_name)   

    
    def get_skip_traces(self, trace_str):
        if trace_str == 'None':
            return "None"

        else:
            traces = trace_str.split("{")[-1].split("}")[0].split(",")
            return [int(a) for a in traces]
        
            
    
# exp_list_and_paths = {
    
    
#     '8mm_GORE_demo' : (Experiment, data_dir + 'GORE_8mm_demo', 'GORE_graft_demo_test2.txt'),
    
    
#     }