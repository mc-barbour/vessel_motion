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

"""
processing sandbox script for the PU results aquired in May 2022.

Each experiment is defined by a class object.
Processing algorthms are defined as functions in source/processing.py

"""

#%% get experiments names for all PU

exp_names = get_experiment_names()
pu_exp_names = []

for name in exp_names:
    if "PU_" in name:
        pu_exp_names.append(name)

#%% remove a few of the experiments
bad_runs = ['PU_03_50_150_RT_v3', 'PU_03_50_150_RT_v4', 'PU_03_50_90_RT_v1']
for run in bad_runs:
    pu_exp_names.remove(run)

#%% Load all PU experiments
pu_experiments = []
for name in pu_exp_names:
    print(name)
    pu_experiments.append(get_experiment(name))
    
#%% Process the data

for exp in pu_experiments[4::]:

    diameters = analyze_image_stack_windows(exp)
    exp.save_diameters(diameters)
    

#%% Plot the data

for exp in pu_experiments[0:8]:
    diameters = exp.load_diameter()
    exp.load_pressure()
    
    dt = 1./ exp.frequency

    
    time = np.arange(len(diameters))*dt

    n_stations = ['station' in a for a in diameters.columns.values].count(True)

    fig = make_subplots(rows=2, cols=1)
    for n in range(n_stations):
        fig.add_trace(go.Scatter(x=time,y=diameters['station '+str(n+1)], name='station ' + str(n+1)), row=1, col=1)
    fig.update_yaxes(title='Diameter (pixels)', row=1, col=1)

    
    fig.add_trace(go.Scatter(x=time,y=exp.pressure_data[exp.start_image:exp.last_image]), row=2, col=1)
    fig.update_yaxes(title='Pressure (mmHg)', row=2, col=1)
    fig.update_layout(title=exp.get_name())
    fig.update_xaxes(title='time (s)')
    fig.show()
    

#%% figure out the peak and min search algorithm
import scipy.signal
search_window=80
exp=pu_experiments[4]
diameters = exp.load_diameter()
exp.load_pressure()


time = np.arange(len(diameters))*dt

n_stations = ['station' in a for a in diameters.columns.values].count(True)


pressure = exp.pressure_data[exp.start_image:exp.last_image].values


peak_vals = scipy.signal.find_peaks(pressure, distance=search_window)
min_vals = scipy.signal.find_peaks(pressure*-1, distance=search_window)


fig = make_subplots(rows=2, cols=1)
for n in range(n_stations):
    fig.add_trace(go.Scatter(x=time,y=diameters['station '+str(n+1)]*exp.scale, name='station ' + str(n+1)), row=1, col=1)
    peak_vals = scipy.signal.find_peaks(diameters['station '+str(n+1)], distance=search_window)
    min_vals = scipy.signal.find_peaks(diameters['station '+str(n+1)]*-1, distance=search_window)
    fig.add_trace(go.Scatter(x=time[peak_vals[0]],y=diameters['station '+str(n+1)][peak_vals[0]]*exp.scale, mode='markers', name='station ' + str(n+1), marker_color='red'), row=1, col=1)
    fig.add_trace(go.Scatter(x=time[min_vals[0]],y=diameters['station '+str(n+1)][min_vals[0]]*exp.scale, mode='markers', name='station ' + str(n+1), marker_color='red'), row=1, col=1)
    fig.update_yaxes(title='Diameter (mm)', row=1, col=1)
fig.update_xaxes(range=[0,10], row=1, col=1, showline=True)


peak_vals = scipy.signal.find_peaks(pressure, distance=search_window)
min_vals = scipy.signal.find_peaks(pressure*-1, distance=search_window)

fig.add_trace(go.Scatter(x=time, y=pressure), row=2, col=1)
fig.add_trace(go.Scatter(x=time[peak_vals[0]],y=pressure[peak_vals[0]], mode='markers'),row=2,col=1)
fig.add_trace(go.Scatter(x=time[min_vals[0]],y=pressure[min_vals[0]], mode='markers', marker_color='red'),row=2,col=1)
fig.update_yaxes(title='Pressure (mmHg)', row=2, col=1)
fig.update_xaxes(title='Time (s)', range=[0,10], row=2, col=1, showline=True)
fig.update_layout(template='plotly_white')
# fig.update_xaxes(range=[0,time[-1]])
fig.show()


#%% sandbox for processing algorithms. Final version moved to processing script

def waveform_amplitude(data, minimums, maximums):
    
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



def compute_compliance(exp, search_window=100, show_plot=True):
    
    
    pressure = exp.pressure_data[exp.start_image:exp.last_image]
    df_diameter = exp.load_diameter()
    
    peak_vals_pressure = scipy.signal.find_peaks(pressure, distance=search_window)[0]
    min_vals_pressure = scipy.signal.find_peaks(pressure*-1, distance=search_window)[0]
    
    pressure_amplitudes = waveform_amplitude(pressure, min_vals_pressure, peak_vals_pressure)
    
    df_amplitudes = pd.DataFrame(data=pressure_amplitudes, columns=['Pressure (mmHg'])
    
    n_stations = ['station' in a for a in diameters.columns.values].count(True)
    peak_vals_all = []
    min_vals_all = []
    for count in range(n_stations):
        
        peak_vals = scipy.signal.find_peaks(df_diameter['station '+str(count+1)], distance=search_window)[0]
        min_vals = scipy.signal.find_peaks(df_diameter['station '+str(count+1)]*-1, distance=search_window)[0]
        
        diameter_amplitude = waveform_amplitude(df_diameter['station '+str(count+1)],min_vals, peak_vals)
        diameter_amplitudes_scaled = [a * exp.scale for a in diameter_amplitude]
        
        df_amplitudes["Diameter, Station "+str(count)] = diameter_amplitudes_scaled       
        
        peak_vals_all.append(peak_vals)
        min_vals_all.append(min_vals)
    
    if show_plot:
        
        fig = make_subplots(rows=2, cols=1)
        for n in range(n_stations):
            fig.add_trace(go.Scatter(x=time,y=df_diameter['station '+str(n+1)], name='station ' + str(n+1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=time[peak_vals_all[n]],y=df_diameter['station '+str(n+1)][peak_vals_all[n]],mode='markers', name='station ' + str(n+1), marker_color='red'), row=1, col=1)
            fig.add_trace(go.Scatter(x=time[min_vals_all[n]],y=df_diameter['station '+str(n+1)][min_vals_all[n]],mode='markers', name='station ' + str(n+1), marker_color='blue'), row=1, col=1)
            fig.update_yaxes(title='Diameter (pixels)', row=1, col=1)


        fig.add_trace(go.Scatter(x=time, y=pressure[exp.start_image:exp.last_image]), row=2, col=1)
        fig.add_trace(go.Scatter(x=time[peak_vals_pressure],y=pressure[peak_vals_pressure], mode='markers'),row=2,col=1)
        fig.add_trace(go.Scatter(x=time[min_vals_pressure],y=pressure[min_vals_pressure], mode='markers', marker_color='green'),row=2,col=1)
        fig.update_yaxes(title='Pressure (mmHg)', row=2, col=1)
        fig.update_xaxes(title='time (s)')

        fig.update_xaxes(range=[0,time[-1]])
        fig.show()
        
    return df_amplitudes

    
    
#%% Compute compliance

for exp in pu_experiments:
    print(exp.get_name())
    df_amp = compute_compliance(exp, search_window=75)
    exp.save_compliance(df_amp, OVERWRITE=False)

# ugh uneven lengths of arrays appending to the is screwing me up... until next time
    
    
#%% create version for debugging individual 
search_window=75
show_plot=True


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
    
    # peak_vals = peak_vals[1:-1]
    # min_vals = min_vals[1:-1]
    
    print(min_vals,peak_vals)
    
    diameter_amplitude = waveform_amplitude(df_diameter['station '+str(count+1)],min_vals, peak_vals)
    diameter_amplitudes_scaled = [a * exp.scale for a in diameter_amplitude]
    
    df_temp = pd.DataFrame(data = diameter_amplitudes_scaled, columns = ["Diameter, Station "+str(count)])    
    
    peak_vals_all.append(peak_vals)
    min_vals_all.append(min_vals)
    
    df_amplitudes = pd.concat([df_amplitudes,df_temp],axis=1)

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

    fig.update_xaxes(range=[0,time[-1]])
    fig.show()

#%% Let's plot all of the amplitudes in time


for exp in pu_experiments:
    fig = go.Figure()
    amp = exp.load_compliance()
    
    if exp.skip_traces != "None":
        bad_trace = exp.skip_traces
        bad_trace_names = ["Diameter, Station " + str(a) for a in bad_trace]
        amp.drop(columns=bad_trace_names, inplace=True)
    index = ["Diameter" in string for string in amp.columns.values]
    trace_names = amp.columns.values[index]
    for name in trace_names:
        fig.add_trace(go.Box(y=amp[name] / amp["Pressure (mmHg"] * 1000, boxpoints='all'))
        fig.update_layout(title=exp.get_name())
    
    fig.show()
    
    
#%% get all graft and experiment names - no version
pu_names_combined = [string[0:-3] for string in pu_exp_names]
pu_names_combined = np.unique(pu_names_combined)

#%% Rewrite amplitdues - 6 had the wrong scaling (off by a factor of ten)


names = ['PU_02_80_120_RT_v1', 'PU_02_110_150_RT_v1', 'PU_02_50_90_RT_v1','PU_02_80_120_RT_v2', 'PU_02_110_150_RT_v2', 'PU_02_50_90_RT_v2']

for name in names:
    exp = get_experiment(name)
    amp = exp.load_amplitudes()
    index = ["Diameter" in string for string in amp.columns.values]
    trace_names = amp.columns.values[index]
    for column in trace_names:
        amp[column] = amp[column] / 10.

    exp.save_amplitudes(amp, OVERWRITE=True)

#%% create one box plot for each exp type

plot_names = ['PU_02_50_90_RT', 'PU_02_80_120_RT', 'PU_02_110_150_RT', 'PU_02_50_150_RT', 
              'PU_03_50_90_RT', 'PU_03_80_120_RT', 'PU_03_110_150_RT', 'PU_03_50_150_RT',
              'PU_04_50_90_RT', 'PU_04_80_120_RT', 'PU_04_110_150_RT', 'PU_04_50_150_RT']


n_plots = len(plot_names)

colors = ['blue', 'red', 'green', 'black']
n_colors = len(colors)

fig = go.Figure()
for count, pu_name in enumerate(plot_names):
    
    plot_count = count - int(count / n_colors)*n_colors
    
    exp_names = [pu_name in string for string in pu_exp_names]
    exp_names = np.array(pu_exp_names)[exp_names]
    
    compliance = np.array([])
    for exp_name in exp_names:
        
        
        exp = get_experiment(exp_name)
        amp = exp.load_compliance()
        
        if exp.skip_traces != "None":
            print("Dropping bad traces")
            bad_trace = exp.skip_traces
            bad_trace_names = ["Diameter, Station " + str(a) for a in bad_trace]
            amp.drop(columns=bad_trace_names, inplace=True)
        
        
        index = ["Diameter" in string for string in amp.columns.values]
        trace_names = amp.columns.values[index]
        for column in trace_names:
            compliance = np.append(compliance, (amp[column].values / amp["Pressure (mmHg"].values) * 10000)
            
        
    fig.add_trace(go.Box(y = compliance, boxpoints='all', name = pu_name, marker_color=colors[plot_count]))
fig.update_yaxes(title='Compliance (% / mmHg)')
fig.show()
        

#%% Process the gore results
gore_exp_names = ['GORE_6mmID_50_150_v2','GORE2_6mmID_50_150_v1','GORE2_6mmID_50_150_v2']

gore_exp_names = ['GORE2_6mmID_50_150_v2']

for name in gore_exp_names:
    exp = get_experiment(name)
    diameters = analyze_image_stack_windows(exp)
    exp.save_diameters(diameters)
    
#%%    
gore_exp_names = ['GORE_6mmID_50_150_v1','GORE_6mmID_50_150_v2','GORE2_6mmID_50_150_v1','GORE2_6mmID_50_150_v2']

gore_exp_names = ['GORE_6mmID_50_150_v2','GORE2_6mmID_50_150_v1']
stations = [[1,2,5], [4,5]]

for count,name in enumerate(gore_exp_names):
    exp = get_experiment(name)
    df_diameter = exp.load_diameter()
    exp.load_pressure()
    pressure = exp.pressure_data
    
    n_stations = ['station' in a for a in df_diameter.columns.values].count(True)
    
    
    dt = 1./ exp.frequency
    time = np.arange(len(df_diameter))*dt
    fig = make_subplots(rows=2, cols=1)
    for n in range(n_stations):
        fig.add_trace(go.Scatter(x=time,y=df_diameter['station '+str(n+1)], name='station ' + str(n+1)), row=1, col=1)
        # fig.add_trace(go.Scatter(x=time[peak_vals_all[n]],y=df_diameter['station '+str(n+1)][peak_vals_all[n]],mode='markers', name='station ' + str(n+1), marker_color='red'), row=1, col=1)
        # fig.add_trace(go.Scatter(x=time[min_vals_all[n]],y=df_diameter['station '+str(n+1)][min_vals_all[n]],mode='markers', name='station ' + str(n+1), marker_color='blue'), row=1, col=1)
        # fig.update_yaxes(title='Diameter (pixels)', row=1, col=1)


    fig.add_trace(go.Scatter(x=time, y=pressure), row=2, col=1)
    # fig.add_trace(go.Scatter(x=time[peak_vals_pressure],y=pressure[peak_vals_pressure], mode='markers'),row=2,col=1)
    # fig.add_trace(go.Scatter(x=time[min_vals_pressure],y=pressure[min_vals_pressure], mode='markers', marker_color='green'),row=2,col=1)
    fig.update_yaxes(title='Pressure (mmHg)', row=2, col=1)
    fig.update_xaxes(title='time (s)')

    fig.update_xaxes(range=[0,time[-1]])
    fig.show()
    
    
    
    df_amp = compute_compliance_stations(exp, stations[count], search_window=90)
    exp.save_compliance(df_amp, OVERWRITE=True)
    
    
#%% plot the GORE graft amplitudes
gore_exp_names = ['GORE_6mmID_50_150_v2','GORE2_6mmID_50_150_v1']

compliance_gore = np.array([])
for exp_name in gore_exp_names:
    exp = get_experiment(exp_name)
    amp = exp.load_compliance()
    
    index = ["Diameter" in string for string in amp.columns.values]
    trace_names = amp.columns.values[index]
    for column in trace_names:
        compliance_gore = np.append(compliance_gore, amp[column].values / amp["Pressure (mmHg"].values * 1000)

fig = go.Figure()
fig.add_trace(go.Box(y=compliance_gore))
fig.show()


#%% function for computing a single value of compliance

exp = get_experiment("PU_03_80_120_RT_v2")
    
for exp in pu_experiments:
    df_amp = exp.load_amplitudes()
    compliance = average_compliance(exp)
    
    
    
    
    
def average_compliance(exp):
    
    
    
 #   
    
#%% Plot just the small ranges, compliance
plot_names = ['PU_02_50_90_RT', 'PU_02_80_120_RT', 'PU_02_110_150_RT', 
              'PU_03_50_90_RT', 'PU_03_80_120_RT', 'PU_03_110_150_RT',
              'PU_04_50_90_RT', 'PU_04_80_120_RT', 'PU_04_110_150_RT',]


n_plots = len(plot_names)

colors = ['blue', 'red', 'green']
n_colors = len(colors)

fig = go.Figure()
for count, pu_name in enumerate(plot_names):
    
    plot_count = count - int(count / n_colors)*n_colors
    plot_count = int(count / n_colors)
    
    exp_names = [pu_name in string for string in pu_exp_names]
    exp_names = np.array(pu_exp_names)[exp_names]
    
    compliance = np.array([])
    for exp_name in exp_names:
        
        
        exp = get_experiment(exp_name)
        amp = exp.load_compliance()
        
        if exp.skip_traces != "None":
            print("Dropping bad traces")
            bad_trace = exp.skip_traces
            bad_trace_names = ["Diameter, Station " + str(a) for a in bad_trace]
            amp.drop(columns=bad_trace_names, inplace=True)
        
        
        index = ["Diameter" in string for string in amp.columns.values]
        trace_names = amp.columns.values[index]
        for column in trace_names:
            compliance = np.append(compliance, (amp[column].values / amp["Pressure (mmHg"].values) * 10000)
            
        
    fig.add_trace(go.Box(y = compliance, name = pu_name, marker_color=colors[plot_count]))


fig.update_yaxes(title='Compliance (% / 100 mmHg)')
fig.update_xaxes(title = "Dynamic Pressure Range (mmHg)", ticktext=["50 - 90", "80 - 120", "110 - 150", "50 - 90", "80 - 120", "110 - 150","50 - 90", "80 - 120", "110 - 150" ], tickvals=plot_names )


fig.show()
        
    
#%% Plot just the small ranges, amplitudes
plot_names = ['PU_02_50_90_RT', 'PU_02_80_120_RT', 'PU_02_110_150_RT', 
              'PU_03_50_90_RT', 'PU_03_80_120_RT', 'PU_03_110_150_RT',
              'PU_04_50_90_RT', 'PU_04_80_120_RT', 'PU_04_110_150_RT',]


n_plots = len(plot_names)

colors = ['blue', 'red', 'green']
n_colors = len(colors)

fig = go.Figure()
for count, pu_name in enumerate(plot_names):
    
    plot_count = count - int(count / n_colors)*n_colors
    
    exp_names = [pu_name in string for string in pu_exp_names]
    exp_names = np.array(pu_exp_names)[exp_names]
    
    compliance = np.array([])
    for exp_name in exp_names:
        
        
        exp = get_experiment(exp_name)
        amp = exp.load_amplitudes()
        
        if exp.skip_traces != "None":
            print("Dropping bad traces", exp_name)
            bad_trace = exp.skip_traces
            bad_trace_names = ["Diameter, Station " + str(a) for a in bad_trace]
            amp.drop(columns=bad_trace_names, inplace=True)
        
        
        index = ["Diameter" in string for string in amp.columns.values]
        trace_names = amp.columns.values[index]
        for column in trace_names:
            compliance = np.append(compliance, (amp[column].values / amp["Pressure (mmHg"].values) * 10000)
            
        
    fig.add_trace(go.Box(y = compliance, name = pu_name, marker_color=colors[plot_count]))
fig.update_yaxes(title='Compliance (% / 100 mmHg)')
fig.show() 
 
 
#%% 
plot_names = ['PU_02_50_150_RT', 'PU_03_50_150_RT', 'PU_04_50_150_RT']    
    
n_plots = len(plot_names)

colors = ['blue', 'red', 'green', 'black']
n_colors = len(colors)

fig = go.Figure()
for count, pu_name in enumerate(plot_names):
    
    plot_count = count - int(count / n_colors)*n_colors
    
    exp_names = [pu_name in string for string in pu_exp_names]
    exp_names = np.array(pu_exp_names)[exp_names]
    
    compliance = np.array([])
    for exp_name in exp_names:
        
        
        exp = get_experiment(exp_name)
        amp = exp.load_compliance()
        
        if exp.skip_traces != "None":
            print("Dropping bad traces", exp_name)
            bad_trace = exp.skip_traces
            bad_trace_names = ["Diameter, Station " + str(a) for a in bad_trace]
            amp.drop(columns=bad_trace_names, inplace=True)
        
        
        index = ["Diameter" in string for string in amp.columns.values]
        trace_names = amp.columns.values[index]
        for column in trace_names:
            compliance = np.append(compliance, (amp[column].values / amp["Pressure (mmHg"].values) * 100 * 100)
            
        
    fig.add_trace(go.Box(y = compliance, name = pu_name, marker_color=colors[count]))
fig.add_trace(go.Box(y = compliance_gore, name = "GORE", marker_color='black'))
fig.update_yaxes(title='Compliance (% / 100 mmHg )')
fig.update_layout(title='Dynamic Range: 50 - 150 mmHg')
fig.update_xaxes(title = "Tested Graft",ticktext=["PU 02", "PU 03", "PU 04", "GORE, 6mm"], tickvals=['PU_02_50_150_RT', 'PU_03_50_150_RT', 'PU_04_50_150_RT', "GORE"] )
fig.show() 
    
    
    
    
    
#%% Plot just the small ranges, compliance - Just for a single graft - PU02
plot_names = ['PU_02_50_90_RT', 'PU_02_80_120_RT', 'PU_02_110_150_RT']


n_plots = len(plot_names)

colors = ['blue', 'red', 'green']
n_colors = len(colors)

fig = go.Figure()
for count, pu_name in enumerate(plot_names):
    
    plot_count = count - int(count / n_colors)*n_colors
    plot_count = int(count / n_colors)
    
    exp_names = [pu_name in string for string in pu_exp_names]
    exp_names = np.array(pu_exp_names)[exp_names]
    
    compliance = np.array([])
    for exp_name in exp_names:
        
        
        exp = get_experiment(exp_name)
        amp = exp.load_compliance()
        
        if exp.skip_traces != "None":
            print("Dropping bad traces")
            bad_trace = exp.skip_traces
            bad_trace_names = ["Diameter, Station " + str(a) for a in bad_trace]
            amp.drop(columns=bad_trace_names, inplace=True)
        
        
        index = ["Diameter" in string for string in amp.columns.values]
        trace_names = amp.columns.values[index]
        for column in trace_names:
            compliance = np.append(compliance, (amp[column].values / amp["Pressure (mmHg"].values) * 10000)
            
        
    fig.add_trace(go.Box(y = compliance, name = pu_name, marker_color=colors[plot_count]))


fig.update_yaxes(title='Compliance (% / 100 mmHg)')
fig.update_xaxes(title = "Dynamic Pressure Range (mmHg)", ticktext=["50 - 90", "80 - 120", "110 - 150", "50 - 90", "80 - 120", "110 - 150","50 - 90", "80 - 120", "110 - 150" ], 
                 tickvals=plot_names)
fig.update_layout(template='plotly_white')

fig.show()
        


#%% 50 - 150 dynamic range plot

plot_names = ['PU_02_50_150_RT', 'PU_03_50_150_RT', 'PU_04_50_150_RT']    
    
n_plots = len(plot_names)

colors = ['blue', 'red', 'green', 'black']
n_colors = len(colors)

fig = go.Figure()
compliance = np.array([])
for count, pu_name in enumerate(plot_names):
    
    plot_count = count - int(count / n_colors)*n_colors
    
    exp_names = [pu_name in string for string in pu_exp_names]
    exp_names = np.array(pu_exp_names)[exp_names]
    

    for exp_name in exp_names:
        
        
        exp = get_experiment(exp_name)
        amp = exp.load_compliance()
        
        if exp.skip_traces != "None":
            print("Dropping bad traces", exp_name)
            bad_trace = exp.skip_traces
            bad_trace_names = ["Diameter, Station " + str(a) for a in bad_trace]
            amp.drop(columns=bad_trace_names, inplace=True)
        
        
        index = ["Diameter" in string for string in amp.columns.values]
        trace_names = amp.columns.values[index]
        for column in trace_names:
            compliance = np.append(compliance, (amp[column].values / amp["Pressure (mmHg"].values) * 100 * 100)
            
        
fig.add_trace(go.Box(y = compliance, name = pu_name, marker_color='blue'))
fig.add_trace(go.Box(y = compliance_gore, name = "GORE", marker_color='black'))
fig.update_yaxes(title='Compliance (% / 100 mmHg )', type='log')
fig.update_layout(title='Dynamic Range: 50 - 150 mmHg')
# fig.update_xaxes(title = "Tested Graft",ticktext=["PU 02", "PU 03", "PU 04", "GORE, 6mm"], tickvals=['PU_02_50_150_RT', 'PU_03_50_150_RT', 'PU_04_50_150_RT', "GORE"] )
fig.update_xaxes(title = "Dynamic Range: 50-150 mmHg",ticktext=["PU", "GORE"], tickvals=['PU_04_50_150_RT', "GORE"] )
fig.update_layout(template='plotly_white')

fig.show() 
    
       
        
    
    
    
    
    
    
    















    
    