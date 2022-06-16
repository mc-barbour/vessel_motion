#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 10:26:52 2022

@author: mbarbour

Define paths for your local machine.

"""


from pathlib import Path
import platform
import os

mylogin = os.getlogin();
myplatform = platform.sys.platform;


if myplatform == 'win32':
    base_dir = 'C:/Users/MichaelBarbour/OneDrive - UW/GraftTesting'
    
elif myplatform == 'darwin':
    base_dir = '/Users/mbarbour/OneDrive - UW/GraftTesting'

    
data_dir = base_dir + '/Data/'
figure_dir = base_dir + '/Figures/'
