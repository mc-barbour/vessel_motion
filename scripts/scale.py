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
from paths import *


pio.renderers.default = "browser"
#%% set-up directories

scale_dirs = glob.glob(data_dir + '/Scale/*')

#%% loop through directories 


for scale in scale_dirs:
    image_files = glob.glob(scale + "/*.tif")
    first_image = io.imread(image_files[0])
    midpoint = int(first_image.shape[0]/2)
    
    measure_points = [25,50,75,100]
    shaft_diameter = []
    
    for point in measure_points:
        distance, top, bottom = measure_distance_sobel_horizontal(first_image, point, midpoint)
        shaft_diameter.append(distance)
    
    print(scale, np.mean(shaft_diameter), shaft_diameter)

