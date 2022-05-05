#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 10:22:24 2022

@author: mbarbour
"""


import pandas as pd
import glob

from skimage import io


from paths import data_dir


# this isnt called on import...
experiment_log = pd.read_csv(data_dir + 'experiment_log.csv', index_col='name')

def get_experiment(name):
    """Return a single experiment class if available in list of experiments"""
    
    if name in experiment_log.index.values:
        return Experiment(name, experiment_log.loc[name])
    else:
        raise ValueError("No experiment found with name {:s}. Check spelling and paths in experiment.py".format(name))
        
def get_experiment_names():
    """"Print all available experiments"""
    for name in experiment_log.index.values:
        print("{:s}".format(name))



class Experiment():
    
    def __init__(self, name, params):
        
        self._name = name 
        self._path = data_dir + params['data folder name']
        self.pressure_file = self._path + "/" + params['pressure file']
        self.frequency = params['frequency (Hz)']
        self.image_files = self.get_image_files()
        
        self.set_scale(params['scale (mm/pixel)'])
        


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
        

    def read_pressure(self):
        df = pd.read_csv(self.pressure_file, names=('Index', 'Pressure (mmHg)'))
        self.pressure_data = df["Pressure (mmHg)"]
        
    

    
    
# exp_list_and_paths = {
    
    
#     '8mm_GORE_demo' : (Experiment, data_dir + 'GORE_8mm_demo', 'GORE_graft_demo_test2.txt'),
    
    
#     }