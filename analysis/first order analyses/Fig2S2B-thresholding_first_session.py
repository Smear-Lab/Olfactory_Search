'''
Figure 2S2B -- First session of thresholding experiment 

This code calculates and plots the performance between conditions in the first session of thresholding experiments. 

Written by: Teresa Findley, tmfindley15@gmail.com
Last Updated: 04.27.2021
'''

#Import Libraries
from __future__ import division
import numpy as np
import os,sys
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.signal
import scipy.stats as spystats

### --- USER INPUT --- ###

#EXPERIMENT INFO
datadir = "C:/Users/tfindley/Dropbox (University of Oregon)/Matt & Reese - Shared/Documents/Manuscripts/Reese_Manuscript_Search1/dryad/Findley files/"
conc_list = [1, 2, 3] #1 = 90:30, 2 = 30:10, 3 = 0:0
colorlist = ['m','g','k']

#ANALYSIS PREFERENCES

#PLOTTING PREFERENCES
point_size = 10 #size of points
line_width = 1 #width of lines for individual mice
average_line_width = 2.5 #line width for average across mice plots
offset =  0.005; space_between_conditions = 1.5;

#SAVING PREFERENCES
savedir = "C:/Users/tfindley/Dropbox/analysis/general_stats/acrossmice/thresholding/"
graphname = 'thresholding_across_first_session'
filetype = '.svg'
save_plot = False #option to save figure
show_plot = True #option to display plot 

### --- END USER INPUT --- ###

###  FUNCTIONS  ###
'''
Rolling Average over dataset
Input: data array, window size
Output: smoothed data array
'''
def rolling_average(data,rolling_window):
    data = np.convolve(data, np.ones((rolling_window,))/rolling_window, mode='same')
    return data

###  ---------  ###

#Check for Saving Folder
if (os.path.exists(savedir) == False):
    os.makedirs(savedir)
    print("Created Saving Folder") 
else: print ("Saving Folder Exists")

#Initiate Figure
mpl.rcParams['savefig.pad_inches'] = 0
fig = plt.figure(); ax = fig.add_subplot(111)

counter = 0; #initiate variables 

#Find subjects in data directory 
subject_list = os.listdir(datadir) #walk through data directory for list of subjects

ninety = np.zeros((0,200)); ninety.fill(np.nan) #set up numpy arrays to store data in 
thirty = np.zeros((0,200)); thirty.fill(np.nan)

for concentration in conc_list: #loop through 90:30, 30:10, and 0:0
    
    data_array = np.zeros((0,200)); data_array.fill(np.nan) #initiate data array for current concentration condition 
    
    for mouse_id in subject_list: #loop through all mice 
        add_line = np.zeros((1,200)); add_line.fill(np.nan)
        subject_dir = datadir + mouse_id + "/thresholding/"

        if os.path.exists(subject_dir) == False: #skip mouse if there are no thresholding experiments 
            print mouse_id + ': No thresholding experiments'
            continue
        
        measurement_values = [] #list to store individual performance values in 
        
        os.chdir(subject_dir) #navigate to local directory
        session_list = [name for name in os.listdir(".") if os.path.isdir(name)] #find all session folders in working directory
        
        session_dir = subject_dir + session_list[0] + '/' #only take data from first session
        
        if os.path.exists(session_dir) == False: #if there is no session 1, skip this mouse 
            print mouse_id + ': No first session'
            continue
        
        print concentration, mouse_id, session_list[0] #report working directory 
        trialsummaryfile = session_dir + "trial_params_wITI.txt"
        sniff_file = session_dir + 'sniff.bin'
        framesummaryfile = session_dir + "frame_params_wITI.txt"
        
        #Load in data
        trial_summ = np.genfromtxt(trialsummaryfile, delimiter = ',', skip_header = 1)
        frame_summ = np.genfromtxt(framesummaryfile, delimiter = ',', skip_header = 1) 
        
        concsetting = trial_summ[:,0]; trialtype = trial_summ[:,1]; #trial number, concentration setting, left/right
        answer = trial_summ[:,2]; tstart = trial_summ[:,3]; tend = trial_summ[:,4] #correct/incorrect, trial start time, trial end time 

        #do not use sessions with less than 80 trials 
        if len(trial_summ) < 80:
            if error_report == True:
                print mouse_id, 'Session less than 80 trials'
            continue
        
        performance_controls = 0
        if len(answer[concsetting == 3]) > 0: 
            performance_controls = np.sum(answer[concsetting == 3])/len(answer[concsetting == 3])
        if performance_controls >= 0.6: #we exclude high controls, because these indicate there was odor contamination (olfactometers were cleaned directly after these sessions) 
            print mouse_id + ': ' + ': ' + str(session) + ': High Controls'
            continue

        overall_performance = 0; overall_trials = 0 
        #Loop through all trials
        no_trials = len(trial_summ)
        if no_trials > 200: #don't use beyond 200 trials 
            no_trials = 200
        counting_trials = 0
        for current_trial in range (0,no_trials):

            if concsetting[current_trial] == concentration: #make a list of aggregate performance over session 
                overall_performance = overall_performance + answer[current_trial]
                overall_trials = overall_trials + 1
                calc_perf = (overall_performance/overall_trials)*100           
                add_line[0,counting_trials] = calc_perf #add_line is a list of no_trials values that demonstrate performance over session 
                counting_trials = counting_trials + 1
            
        add_line[0,:] = rolling_average(add_line[0,:],20)
        data_array = np.append(data_array, add_line, axis = 0) #create array of total data for mouse 
        if concentration == 1: 
            ninety = np.append(ninety, add_line, axis = 0)
        if concentration == 2:
            thirty = np.append(thirty, add_line, axis = 0)
        counter = counter + 1
        
    means = np.zeros((2,200)); means.fill(np.nan) #calculate mean performance and standard deviation across mice 
    for x in range(0,200):
        means[0,x] = np.nanmean(data_array[:,x])
        means[1,x] = np.nanstd(data_array[:,x])
    
             
    x_values = np.arange(0,200,1) #x values for plot 
    plt.plot(x_values, means[0,:], linewidth = line_width + 3, color = colorlist[concentration-1]) #plot means 
    plt.fill_between(x_values,means[0,:]-means[1,:],means[0,:]+means[1,:], color = colorlist[concentration-1], alpha = 0.2) #shade in standard deviation
    plt.xlim(10,80); plt.ylim(0,100) #plot limits 
    if show_plot == True:
        plt.pause(.2) 

wilcox = np.zeros((1,200)); wilcox.fill(np.nan) #compare difference between 90:30 and 30:10 performance across all mice 
for x in range(0,200):
    wilcox[0,x] = spystats.wilcoxon(ninety[:,x], thirty[:,x])[1]

print wilcox
if save_plot == True: 
    plt.savefig(savedir + str(session) + graphname + filetype)
if show_plot == True: 
    plt.show() 
