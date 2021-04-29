'''
Figure 2S2A -- Novel odorant 

This code compares olfactory search performance on trained odorant versus the first session of a novel odorant. 
We chose vanillin as a novel odorant, because it does not activate the trigeminal system. 

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
# 

### --- USER INPUT --- ###

#EXPERIMENT INFO
datadir = "C:/Users/tfindley/Dropbox (University of Oregon)/Matt & Reese - Shared/Documents/Manuscripts/Reese_Manuscript_Search1/dryad/Findley files/"

measurement = 'mean_performance' #best_performance,mean_performance,trial_duration
concentration_list = ['p','v']

#ANALYSIS PREFERENCES
#General arena coordinates
nosepoke = [1127,354]
error_report = True #print errors when excluding data 

#PLOTTING PREFERENCES
#Font preferences
chosenfont = {'fontname':'Arial'} #standardize font across plots
axes_font_size = 35
offset =  0.03; space_between_conditions = 1.5;

#SAVING PREFERENCES
savedir = "C:/Users/tfindley/Dropbox/analysis/general_stats/acrossmice/vanillin/"
filetype = '.svg'
graphname = 'vanillin_' + measurement
show_plot = True #option to display plot at the end
save_plot = False #option to save plot at the end 

### --- END USER INPUT --- ###

#Check for Saving Folder
if (os.path.exists(savedir) == False):
    os.makedirs(savedir)
    print("Created Saving Folder") 
else: print ("Saving Folder Exists")

#Initiate Figure
mpl.rcParams['savefig.pad_inches'] = 0
fig = plt.figure(); ax = fig.add_subplot(111)

#create lists & arrays for storing data throughout analysis 
mouse_list = np.zeros((0,len(concentration_list))); x_vals = np.arange(space_between_conditions,(len(concentration_list)+1)*space_between_conditions,space_between_conditions)
session_type = 'none'
p_list = []; v_list = []
pxlist = []; vxlist = []

#Loop through individual subjects
counter = 0; marker_type = -1
#Find subjects in data directory 
subject_list = os.listdir(datadir) #walk through data directory for list of subjects 

trial_time_limit = 10

for mouse_id in subject_list:
    add_line = np.zeros((1,len(concentration_list))); add_line.fill(np.nan)
    marker_type = marker_type + 1
    
    add_line = np.zeros((1,len(concentration_list))); add_line.fill(np.nan)
    subject_dir = datadir + mouse_id + "/interleaved/" #all of these mice were run on interleaved in this experiment 

    if os.path.exists(subject_dir) == False:
        if error_report == True:
            print mouse_id  + ': ' + 'No Experiment Directory'
        continue

    os.chdir(subject_dir) #navigate to local directory
    session_list = [name for name in os.listdir(".") if os.path.isdir(name)] #find all sub-folders
    
    conc_counter = 0
    #loop through different conditions. 0 = 100:0, 1 = 80:20, 2 = 60:40, & 3 = 0:0
    for concentration_setting in concentration_list:
        measurement_values = []; best_performance = 0;   
        for session in session_list:
        
            session_dir = subject_dir + str(session) + '/'
        
            print mouse_id, session

            trialsummaryfile = session_dir + "trial_params_wITI.txt" #data files 
            notesfile = session_dir + 'notes.txt' #general session notes

            #data exclusion criteria 
            if os.path.isfile(trialsummaryfile) == False:
                trialsummaryfile = session_dir + "trial_params.txt"
            if os.path.isfile(trialsummaryfile) == False: 
                if error_report == True:
                    print mouse_id + ": " + str(session) + ' No Trial Summary File'
                continue
            
            #Load in data
            trial_summ = np.genfromtxt(trialsummaryfile, delimiter = ',', skip_header = 1) 
            concsetting = trial_summ[:,0]; trialtype = trial_summ[:,1]; #trial number, concentration setting, left/right
            answer = trial_summ[:,2]; tstart = trial_summ[:,3]; tend = trial_summ[:,4] #correct/incorrect, trial start time, trial end time 
            
            #data exclusion criteria 
            if os.path.isfile(notesfile) == False:
                if error_report == True:
                    print mouse_id + ": " + str(session) + ' No Notes File'
                continue
                     
            with open (notesfile, "r") as myfile:
                notes = myfile.readlines()
                for line in range(0,len(notes)):
                    if 'Odor Type' in notes[line]:
                        if 'v' in notes[line].lower():
                            print "VANILLIN" #tell user when a vanillin session is found 
                            session_type = 'v'; break
                        elif 'p' in notes[line].lower():
                            session_type = 'p'; break
            
            if session_type == concentration_setting:

                #exclude sessions with less than 80 trials 
                if len(trial_summ) < 80:
                    if error_report == True:
                        print mouse_id, session, 'Session less than 80 trials'
                    continue
                    
                performance_controls = 0
                if len(answer[concsetting == 3]) > 0: 
                    performance_controls = np.sum(answer[concsetting == 3])/len(answer[concsetting == 3])
                if performance_controls >= 0.6: #exclude sessions with high control performance. We used these sessions as an indicator we needed to 
                    #clean the olfactometers...therefore, they are excluded because the high performance indicates system contamination with odor
                    print mouse_id + ': ' + str(session) + ': High Controls'
                    continue
            
                if measurement == 'best_performance' or measurement == 'mean_performance':
                    performance = np.sum(answer[concsetting == 1])/len(answer[concsetting == 1])*100
                    measurement_values.append(performance)
                    if performance > best_performance:
                        best_performance = performance
         
                #Loop through all trials 
                for current_trial in range (0,len(trial_summ)):
                    start = tstart[current_trial]; end = tend[current_trial];
                    if end - start > trial_time_limit: #set trial time limit
                        if error_report == True:
                            print mouse_id,  session, current_trial, 'Trial too long'
                        continue
                    if measurement == 'trial_duration':
                        trialduration = end - start
                        if concsetting[current_trial] == 1:
                            measurement_values.append(trialduration)
                            if session_type == 'p': p_list.append(trialduration)
                            if session_type == 'v': v_list.append(trialduration)

        #add trial statistics to full array 
        if best_performance == 0:
            best_performance = np.nan                
        if measurement == 'best_performance':
            add_line[0,conc_counter] = best_performance
        if measurement == 'trial_duration' or measurement == 'mean_performance':
            add_line[0,conc_counter] = np.nanmean(measurement_values)

        conc_counter = conc_counter + 1
    if np.isnan(add_line[0,1]) == True:
        continue      
    # add overall trial statistics for each mouse to mouse array 
    mouse_list = np.append(mouse_list,add_line,axis = 0)
    print mouse_id, ':', mouse_list[counter,:]
    counter = counter + 1
print len(mouse_list)

means = np.zeros((2,len(concentration_list)))
for p in range(0,len(concentration_list)):
    means[0,p] = np.nanmean(mouse_list[:,p])
    means[1,p] = np.nanstd(mouse_list[:,p])
print means

x_values = x_vals - (offset*counter/2)

x_axis = np.arange(len(means[0,:]))
x_axis = (x_axis + 1)*1.2

ax.bar(x_axis, means[0,:], color = 'g', alpha = 0.5,yerr = means[1,:], capsize = 10) #plot bar plot of pinene versus vanillin 

#plot settings
if measurement == 'best_performance' or measurement == 'mean_performance':
    ymin = 0; ymax = 100
if measurement == 'trial_duration':
    ymin = 0; ymax = 3

ax.set_ylim(ymin, ymax)

ax.tick_params(top = 'off', right = 'off', bottom = 'off')
ax.spines['top'].set_position(('data',0))
ax.spines['right'].set_position(('data',0))
plt.gca().spines['bottom'].set_position(('data',0))
ax.spines['bottom'].set_visible(False)
plt.gca().axes.get_xaxis().set_visible(False)
plt.yticks(fontsize=axes_font_size)

print mouse_list[:,0]

if save_plot == True:
    plt.savefig(savedir + graphname + filetype)
if show_plot == True: 
    plt.show() 
