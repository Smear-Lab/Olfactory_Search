'''
Figure 1E -- Pressure cannula signal versus Thermistor signal 

This code compares simultaneously recorded pressure and thermistor sniff signals in the same mouse to 
evaluate the precision of thermistor sniff recordings. 

Written by: Teresa Findley, tmfindley15@gmail.com
Last Updated: 04.27.2021
'''

##SET UP

#Imports
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from PyAstronomy import pyaC
import scipy.signal as spysig
import sys

### --- USER INPUT --- ###

directory = 'C:/Users/tfindley/Dropbox (University of Oregon)/Matt & Reese - Shared/Documents/Manuscripts/Reese_Manuscript_Search1/dryad/Findley files/thermistor_vs_pressure_sensor/' #location of saved data 
#Choose what you want to plot 
plot = 1 #0 = example signal comparison plot (F1E-top), 1 = jitter histogram (F1E-bottom left), 2 = sniff freq scatter (F1E-bottom right) 
binwidth = 25 #histogram binwidth 
number_of_recordings = 5 #how many recordings were taken 
display_figure = True #show figure at end or not
save_figure = False #save figure at end or not

### --- END USER INPUT --- ###

###  FUNCTIONS  ### 
def rolling_average(data,rolling_window): #apply a rolling average to a signal 
    data = np.convolve(data, np.ones((rolling_window,))/rolling_window, mode='same')
    return data

def local_maxima(data,peakwindow): #find local maxima in a signal given a window of samples 
    localmaxima = spysig.argrelmax(data,order = peakwindow)
    localmaxima = localmaxima[0]
    return localmaxima

###  ---------  ###

fig = plt.figure(); ax = fig.add_subplot(111) #initiate figures 

sniff_error = []; sniff_freq = [] #initiate lists used to create figure plots 

for session in range(0,number_of_recordings): #loop through data folder   
    #Locate data files 
    ch0_file = directory + '/thermistor' + str(session) + '.bin' #Thermistor sniff signal 
    ch2_file = directory + '/pressure_cannula' + str(session) + '.bin' #Pressure sniff signal 
    
    #Read sniff signals into numpy arrays 
    thermistor = np.fromfile(ch0_file,dtype = 'float')
    pressure = np.fromfile(ch2_file,dtype = 'float')
    
    #Smooth thermistor & pressure sniff signals (for example plot, smooth with window the size of ~10% sampling rate - 800 Hz)  
    if plot == 0: #if plotting example data 
        thermistor = rolling_average(thermistor,81)
    #Smooth both sniff signals with windows the size of ~2.5% sampling rate - 800 Hz 
    else: thermistor = rolling_average(thermistor,21)
    pressure = rolling_average(pressure, 21)
    
    #Normalize signals -- take mean of each signal and subtract from array to center around 0
    avgtherm = np.mean(thermistor)
    avgpressure = np.mean(pressure)
    thermistor = ((thermistor - avgtherm)*70) #multiply thermistor signal array to be closer to the amplitude of pressure signal (only for F1E-top) 
    pressure = (pressure - avgpressure)
    thermistor = thermistor[100:(len(thermistor)-100)] #remove first and last 100 samples from signals
    pressure = pressure[100:(len(pressure)-100)]
    
    #Identify inhalation points in each signal -- thermistor inhalations are at local maxima, pressure inhalations are at zero crossings
    therm_inhales = local_maxima(thermistor, 31) #find peaks in thermistor signal for inhalation points 
    
    x_vals = np.arange(len(pressure)) #create array of x values for pyaC function
    press_crossings = pyaC.zerocross1d(x_vals, pressure) #use function in pyaC library to find zero crossings of pressure signal 
    press_crossings = [round(x) for x in press_crossings]
    press_crossings = [int(x) for x in press_crossings]
    press_inhales = []
    for i in range(0,len(press_crossings)):
        if pressure[press_crossings[i]-10] - pressure[press_crossings[i] + 10] > 0.5: #exclude high amplitude crossings (occasional error in data collection -- verified by visualization of each crossing) 
            press_inhales.append(press_crossings[i]) #create list of pressure signal inhalation points 
    
    #Calculate jitter between 2 signals and the sniff frequency at each inhalation point 
    for i in range(0,len(press_inhales)): #run through each inhalation
        value = min(therm_inhales, key=lambda x:abs(x-press_inhales[i])) #calculate number of samples between pressure and thermistor inhalation point
        sniff_error.append(1000*(press_inhales[i] - value)/800) #convert into milliseconds and save in sniff error list 
        freq = abs((press_inhales[i] - press_inhales[i-1])/800) #calculate sniff frequency at each inhalation using pressure signal
        sniff_freq.append(1/freq) #sniff frequency saved in Hz
    
    #F1E - Top   
    if plot == 0: #example comparison of 2 signals 
        if session == 1: #use first data recording (could be any) 
            plt.plot(thermistor,'r')
            plt.plot(pressure,'b')
            for x in range(0,len(therm_inhales)): #plot thermistor inhalation points 
                plt.scatter(therm_inhales[x],thermistor[therm_inhales[x]], color = 'maroon',zorder = 3)
            for x in range(0,len(press_inhales)): #plot pressure inhalation points 
                plt.scatter(press_inhales[x],pressure[press_inhales[x]], color = 'navy', zorder = 3)
            plt.xlim(8000,10400); plt.ylim(-4,4) #pick an arbitrary section of signal to plot 
            plt.axis('off') #remove axes 
            graphname = 'example_traces' #name of graph for saving 

#Remove jitter points that are beyond 2 standard deviations of the sniff error distribution (visually verified that these are each errors in data collection) 
sniffstd = np.nanstd(sniff_error)*2 #2 standard deviations from sniff error distribution 
delete_lines = []
for i in range(0,len(sniff_error)):
    if abs(sniff_error[i]) >= sniffstd: 
        delete_lines.append(i)
    elif sniff_freq[i] > 20: #if sniff frequency exceeds 20 Hz (higher than mice have been shown to sniff in freely moving or headfixed conditions -- errors visually verified) 
        delete_lines.append(i)
    elif sniff_freq[i] < .2: #errors in data recording visually verified --remove sniff frequencies below .2 Hz 
        delete_lines.append(i)
for index in sorted(delete_lines, reverse=True): #remove error lines from data as identified above 
    del sniff_freq[index]
    del sniff_error[index]

print np.mean(sniff_error), np.median(sniff_error), np.std(sniff_error)

#F1E - bottom left 
if plot == 1: 
    sniff_error = np.asarray(sniff_error) #plot a histogram of the ms timing difference between pressure and sniff signals 
    plt.hist(sniff_error,color = 'purple', bins = binwidth, weights=np.zeros_like(sniff_error) + 1. / sniff_error.size)
    graphname = 'sniff_jitter' #name of graph being saved 
    ax.tick_params(top = 'off', right = 'off') #axes removal settings 
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([]) 
    ax.spines['top'].set_position(('data',0))
    ax.spines['right'].set_position(('data',0))   
    ax.spines['bottom'].set_position(('data',0))

#F1E - bottom right 
if plot == 2: 
    plt.scatter(sniff_error,sniff_freq,color = 'purple', alpha = 0.5) #plot a scatter plot comparing timing difference against sniff frequency 
    graphname = 'sniff_jitter_vs_freq' #name of graph 
    ax.tick_params(top = 'off', right = 'off') #axes removal settings 
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([]) 
    ax.spines['top'].set_position(('data',0))
    ax.spines['right'].set_position(('data',0))   
    ax.spines['bottom'].set_position(('data',0))

if save_figure == True: 
    plt.savefig(directory + graphname + '.svg')
if display_figure == True: 
    plt.show() 
