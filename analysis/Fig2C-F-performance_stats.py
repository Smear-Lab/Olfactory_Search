'''
Figure 2C-F -- Performance, trial duration, & path tortuosity between conditions 

This code calculates and plots basic trial/session statistics for a chosen experiment across mice and plots it.
This should be used primarily for plotting performance between conditions in different experiments 

Written by: Teresa Findley, tmfindley15@gmail.com
Last Updated: 04.27.2021
'''

#Library imports
from __future__ import division
import numpy as np
import os,sys,math
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.signal
import scipy.stats as spystats

### --- USER INPUT --- ###

#EXPERIMENT INFO
datadir = "C:/Users/tfindley/Dropbox (University of Oregon)/Matt & Reese - Shared/Documents/Manuscripts/Reese_Manuscript_Search1/dryad/Findley files/"

measurement = 'trial_duration'; #represents trial/session statistic being calculated 
#possible measurements: mean_performance, trial_duration, tortuosity

exp_type = 'interleaved' #represents experiment type for analysis
#possible types: interleaved, odor_omission, thresholding, occlusion, novel_odor

#ANALYSIS PREFERENCES
error_report = False; #report reason for skipped files/trials/values
control_threshold = 0.6 #threshold of control performance for excluding session
performance_threshold = 0.6 #threshold on 100-0 performance for excluding session 
min_trial_limit = 80; #minimum number of trials to accept a session
trial_time_limit = 10; #maximum time per trial

tracking_smoothing_window = 7; #rolling window for smoothing tracking data
camera_framerate = 80; #framerate in Hz
tracking_jump_threshold = 40; #threshold for jumps in frame to frame nose position (in pixels) 

#PLOT PREFERENCES
#list of colors for each specific mouse (to map the same color to the same subject when running different analyses) 
color_list = [(0.36863,0.3098,0.63529),(0.2857,0.36983,0.68395),(0.23071,0.43015,0.71815),(0.2018,0.49075,0.73722),(0.19848,0.55214,0.73947),(0.23529,0.6196,0.71509),(0.3005,0.68667,0.67839),(0.37336,0.74331,0.65077),(0.44002,0.78249,0.64609),(0.51396,0.81261,0.64521),(0.59089,0.8385,0.64457),(0.66379,0.86399,0.64333),(0.73618,0.89455,0.63513),(0.81107,0.92774,0.61777),(0.87276,0.95314,0.60146),(0.90608,0.96074,0.59705),(0.92228,0.95945,0.62464),(0.93375,0.95657,0.67002),(0.94411,0.95243,0.70581),(0.95705,0.94619,0.70589),(0.97332,0.93181,0.66785),(0.98802,0.91033,0.61079),(0.99582,0.88406,0.55451),(0.99587,0.84909,0.50836),(0.99505,0.79785,0.46152),(0.99372,0.73769,0.41659),(0.99194,0.67629,0.37663),(0.98701,0.60891,0.33523),(0.97783,0.5355,0.29515),(0.96542,0.46486,0.268),(0.94982,0.40523,0.26428),(0.92366,0.35551,0.27786),(0.88838,0.30906,0.29646),(0.84788,0.25954,0.30893),(0.80334,0.20253,0.30792),(0.74965,0.14052,0.29882),(0.68788,0.074178,0.28237),(0.61961,0.0039216,0.25882)]

#Plotting preferences
point_size = 10 #size of points
line_width = 1 #width of lines for individual mice
marker_type = 'o' #type of point
alpha_val = 0.8 #opacity value
average_line_width = 2.5 #line width for average across mice plots
point_edge_width = 0 #outline on points
offset = 0.008; #offset between individual mice in plot
space_between_conditions = 1; #space between x axis ticks on plot

#SAVING PREFERENCES
savedir = 'C:/Users/tfindley/Dropbox/analysis/general_stats/between_conditions/' #saving pathway
filetype = '.svg'
save_plots = False #option to save plots at end or not
show_plots = True; #display plots while saving

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

#set lists and folders for each experiment type 
if exp_type == 'interleaved': 
    exp_condition = [0,1,2]; working_folder = 'interleaved' #0=100:0, 1=80:20, 2=60:40
if exp_type == 'odor_omission': 
    exp_condition = [1,3]; working_folder = '80-20' #1=80:20, 3=0:0
if exp_type == 'thresholding': 
    exp_condition = [1,2]; working_folder = 'thresholding' #1=90:30, 2=30:10
if exp_type == 'occlusion': 
    exp_condition = ['none','sham','stitch']; working_folder = 'nostril-occlusion'
if exp_type == 'novel_odor': 
    exp_condition = ['pinene','vanillin']; working_folder = 'interleaved'

mouse_list = np.zeros((1,len(exp_condition))); mouse_list.fill(np.nan) #set up NaN array for averages for each mouse
x_vals = np.arange(space_between_conditions,(len(exp_condition)+1)*space_between_conditions,space_between_conditions) #set up x axis for plotting individual mice with offset
mouse_counter = 0; #count only analyzed and saved mice 

#Check for plot folder -- if it does not exist, make it 
if (os.path.exists(savedir) == False):
    os.makedirs(savedir)
    print("Created Saving Folder") 
else: print ("Saving Folder Exists")

mpl.rcParams['savefig.pad_inches'] = 0 #initiate figure
fig = plt.figure(); ax = fig.add_subplot(111)

#Use data directory to generate list of experimental subjects 
subject_list = os.listdir(datadir) #walk through data directory for list of subjects 

#Run through each mouse in list of subjects 
for mouse_id in subject_list:
    condition_list = np.zeros((1,len(exp_condition))); condition_list.fill(np.nan) #set up NaN array for individual conditions across sessions
    
    working_dir = datadir + mouse_id + "/" + working_folder + "/" #working directory
    
    #If directory does not exist, report directory skip
    if os.path.exists(working_dir) == False:
        if error_report == True:
            print mouse_id  + ': ' + ': ' + 'No Experiment Directory -- skipping mouse'
        continue
        
    #Walk through individual mouse's directory and create ordered list of sessions 
    os.chdir(working_dir) #navigate to local directory
    session_list = [name for name in os.listdir(".") if os.path.isdir(name)] #find all session folders in working directory
    if exp_type == 'thresholding': #only analyze first session of thresholding experiments
        session_list = session_list[0]  
    session_counter = 0; #count sessions analyzed and plotted
    current_condition = ''; #initiate variable that identifies current condition
    
    #Run through each session in list of all sessions for individual mouse 
    for session in session_list:
        skip_session = False #variable used for exclusion criteria

        session_dir = working_dir + str(session) + '/' #open session directory
        print mouse_id, exp_type, session #report working session to user
        
        trialsummaryfile = session_dir + "trial_params_wITI.txt" #data files 
        sniff_file = session_dir + 'sniff.bin'
        framesummaryfile = session_dir + "frame_params_wITI.txt"
        notesfile = session_dir + 'notes.txt' #general session notes
        
        #Load in data
        trial_summ = np.genfromtxt(trialsummaryfile, delimiter = ',', skip_header = 1)
        frame_summ = np.genfromtxt(framesummaryfile, delimiter = ',', skip_header = 1) 
        
        concsetting = trial_summ[:,0]; trialtype = trial_summ[:,1]; #trial number, concentration setting, left/right
        answer = trial_summ[:,2]; tstart = trial_summ[:,3]; tend = trial_summ[:,4] #correct/incorrect, trial start time, trial end time 
        
        nx = frame_summ[:,0]; ny = frame_summ[:,1]; #nose x and y coordinates (80 hz) 
        hx = frame_summ[:,2]; hy = frame_summ[:,3]; #head x and y coordinates (80 hz) 
        bx = frame_summ[:,4]; by = frame_summ[:,5]; #body x and y coordinates (80 hz) 
        ts = frame_summ[:,6]; tms = frame_summ[:,7]; #timestamp in seconds, timestamp in milliseconds
        
        ##Session Exceptions 
        #if there is no notes file, do not analyze session 
        if exp_type == 'occlusion' or exp_type == 'novel_odor':
            if os.path.isfile(notesfile) == False: 
                if error_report == True: 
                    print mouse_id + ': ' + concentration_setting + ': ' + str(session) + ': No Notes File -- skipping session'
                continue
        
            #Look in notes file and set the current condition for the session if necessary       
            
            with open (notesfile, "r") as myfile:
                notes = myfile.readlines(); skip_session = False
                for line in range(0,len(notes)):
                    #Check odor identity and, if applicable, set current_condition
                    if 'Odor Type' in notes[line]:
                        if 'pine' in notes[line].lower(): #if it is a pinene session, move on
                            current_condition = 'pinene'
                            break
                        elif 'v' in notes[line].lower(): #if it is a vanillin session, only move on if assessing novel odorants
                            current_condition = 'vanillin'
                            if exp_type != 'novel_odor':
                                skip_session = True
                            break
                        else: 
                            skip_session = True; break
                    if exp_type == 'occlusion': #record current conditions for occlusion experiments 
                        if 'Occluded Nostril' in notes[line]:
                            if 'none' in notes[line].lower():
                                current_condition = 'none'; break
                            elif 'sham' in notes[line].lower():
                                current_condition = 'sham'; break
                            elif 'right' in notes[line].lower():
                                current_condition = 'stitch'; break
                            elif 'left' in notes[line].lower():
                                current_condition = 'stitch'; break
        #Skip session if run with another odor than pinene (except in novel odor experiment type) 
        if skip_session == True:
            if error_report == True:
                print "Not run with pinene -- skipping session"
            continue

        #Pre-processing session information
        if len(trial_summ) < min_trial_limit: #if session is less than trial limit
            if error_report == True:
                print mouse_id, session, 'Session does not meet min. trial limit -- skipping session'
            continue
        
        performance_controls = 0 #does control performance exceeds threshold? 
        if len(answer[concsetting == 3]) > 0: 
            performance_controls = np.sum(answer[concsetting == 3])/len(answer[concsetting == 3])
        if performance_controls >= control_threshold: #these were conditions that indicate contamination and were immediately followed by cleaning our olfactometers 
            if error_report == True:
                print mouse_id + ': ' + exp_type + ': ' + str(session) + ': Controls too high -- skipping session'
            continue
        
        if exp_type == 'interleaved' or exp_type == 'odor_omission' or exp_type == 'novel_odor': #is 100:0 performance below threshold? 
            performance_test = np.sum(answer[concsetting == 0])/len(answer[concsetting == 0])
            if performance_test <= performance_threshold:
                if error_report == True:
                    print mouse_id + ': ' + concentration_setting + ': ' + str(session) + ': 100-0 performance too low -- skipping session'
                continue

        #Smoothing tracking data 
        nx = rolling_average(nx,tracking_smoothing_window); ny = rolling_average(ny,tracking_smoothing_window)
        
        #PERFORMANCE
        if measurement == 'mean_performance':
            if exp_type == 'interleaved' or exp_type == 'thresholding' or exp_type == 'odor_omission':
                for condition in exp_condition:
                    index = exp_condition.index(condition)
                    condition_list[len(condition_list[:,0])-1,index]  = (np.sum(answer[concsetting == condition])/len(answer[concsetting == condition]))*100
            
            else: 
                index = exp_condition.index(current_condition) #index of current condition
                condition_list[len(condition_list[:,0])-1,index] = (np.sum(answer[concsetting != 3])/len(answer[concsetting != 3]))*100 #exclude odor omission
                
            #add empty row to condition list 
            add_row = np.zeros((1,len(condition_list[0,:]))); add_row.fill(np.nan)
            condition_list = np.append(condition_list, add_row, axis = 0)
         
        if measurement != 'mean_performance':
            #Run through each trial of a session
            for current_trial in range (0,len(trial_summ)):
                
                if exp_type == 'interleaved' or exp_type == 'thresholding' or exp_type == 'odor_omission': #identify current odor condition
                    current_condition = concsetting[current_trial]
                    if current_condition == 3: #Skip odor omission trials unless anaylzing odor omission 
                        if exp_type != 'odor_omission':
                            if error_report == True:
                                print mouse_id, session, current_trial, 'Odor omission trial -- skipping trial'
                            continue
            
                start = tstart[current_trial]; end = tend[current_trial]; #mark start and end times of trial
                #if trial is too long, move to the next trial
                if end - start > trial_time_limit:
                    if error_report == True:
                        print mouse_id, exp_type, session, current_trial, 'Trial too long -- skipping trial' #error report
                    continue
                
                #TRIAL DURATION
                if measurement == 'trial_duration':
                    trialduration = end - start
                    
                    index = exp_condition.index(current_condition) #index of current condition
                    condition_list[len(condition_list[:,0])-1,index] = trialduration #add trial duration to array
                    
                    #add empty row to condition list 
                    add_row = np.zeros((1,len(condition_list[0,:]))); add_row.fill(np.nan)
                    condition_list = np.append(condition_list, add_row, axis = 0)
                
                #TORTUOSITY     
                if measurement == 'tortuosity':

                    #Find first & last frame of trial in the coordinates arrays using camera frame timestamp file
                    startline = (np.abs(ts - start)).argmin() #find closest tracking timestamp value to trial start time
                    endline = (np.abs(ts - end)).argmin() #find closes tracking timestamp value to trial end time

                    found_crossing = False; #did the mouse cross the decision line? 
                    distance_travelled = 0; #how far has the mouse gone in the trial? 
                    
                    #Run through coordinates from start to end of trial
                    for frames in range(startline,endline+1):
                        if frames == startline:
                            starting_nose_position = [nx[frames],ny[frames]] #save starting position of nose for trial
                        if found_crossing == True:
                            break #if mouse has crossed decision line -- exit loop
                        if nx[frames] <= 421: #if mouse's nose crosses decision line
                            if found_crossing == False:
                                crossing_point = [nx[frames],ny[frames]] #save ending position of nose for trial
                                frame_cross = frames #save which frame marks the decision line
                                found_crossing = True
                                
                        distance = np.sqrt(math.pow((nx[frames+1]-nx[frames]),2) + math.pow((ny[frames+1] - ny[frames]),2)) #find distance between next frame and current frame 
                        if distance >= tracking_jump_threshold: #remove trials with exceedingly unrealistic jumps in tracking 
                            if error_report == True: 
                                print 'Teleporting Mouse -- skipping trial'
                            break #exit trial and do not save 
                        distance_travelled = distance_travelled + distance #add distance between frames to total distance traveled over the trial
                    
                    #after looping through trial, calculate nose tortuosity for entire trial
                    if found_crossing == True:
                        #record distance from starting nose position to nose position while crossing decision point 
                        shortest_distance = np.sqrt(math.pow(abs(crossing_point[0] - starting_nose_position[0]),2) + math.pow(abs(crossing_point[1] - starting_nose_position[1]),2))
                        if shortest_distance < 100: break #cut trials where shortest distance is smaller than possible
                        
                        #total distance travelled/shortest possible distance to calculate tortuosity 
                        index = exp_condition.index(current_condition) #index of current condition
                        condition_list[len(condition_list[:,0])-1,index] = distance_travelled/shortest_distance #add trial duration to array
      
                        #add empty row to condition list 
                        add_row = np.zeros((1,len(condition_list[0,:]))); add_row.fill(np.nan)
                        condition_list = np.append(condition_list, add_row, axis = 0)
        
        #take mean of trial values for a single measurement value per session
        for column in range (0,len(exp_condition)):
            mouse_list[mouse_counter,column] = np.nanmean(condition_list[:,column])
        session_counter = session_counter + 1 #count analyzed and saved session 
    
    x_values = x_vals - (offset*mouse_counter)
    
    if exp_type == 'novel_odor': #for novel odor, only plot mice that have both pinene and vanillin sessions 
        if math.isnan(mouse_list[mouse_counter,1]) == True:
            continue
             
    #Plot session values for current mouse in unique color 
    ax.plot(x_values, mouse_list[mouse_counter,:], color = color_list[subject_list.index(mouse_id)],linewidth = line_width, marker = marker_type,markersize = point_size, markeredgewidth = point_edge_width, alpha = alpha_val)
    if show_plots == True:
        plt.pause(.3)
    mouse_counter = mouse_counter + 1 #count an analyzed and saved mouse
    
    #add a NaN row to mouse array for next mouse
    add_row = np.zeros((1,len(mouse_list[0,:]))); add_row.fill(np.nan)
    mouse_list = np.append(mouse_list, add_row, axis = 0)
    
x_values = x_vals - (offset*mouse_counter/2) #set x axis for plotting average across mice 
mouse_list = (mouse_list[~np.isnan(mouse_list).all(axis=1)]) #remove any rows with all nan values from the mouse_list

if exp_type == 'novel_odor': #remove mice with only pinene and not vanillin from mouse list
    mouse_list = (mouse_list[~np.isnan(mouse_list).any(axis=1)]) #remove any rows with ANY nan values from the mouse_list

#Calculate and print means across mice
means = np.zeros((2,len(mouse_list[0,:])))
for p in range(0,len(mouse_list[0,:])):
    means[0,p] = np.nanmean(mouse_list[:,p])
    means[1,p] = np.nanstd(mouse_list[:,p])
print means

#plot averages across mice with standard deviations 
ax.errorbar(x_values,means[0,:],yerr = means[1,:], linewidth = 0, color = 'k',marker = marker_type,markersize = point_size + 7,elinewidth = average_line_width, zorder = 32)
number = len(mouse_list[:,0]); print number #count and report number of mice used

#Plot settings
if measurement == 'mean_performance': #set ymin and ymax based on measurement type
    ymin = 40; ymax = 100
if measurement == 'trial_duration':
    ymin = 1; ymax = 6
if measurement == 'tortuosity':
    ymin = 1; ymax = 3
xmin = np.min(x_values) - 0.5; xmax = np.max(x_values) + 0.5 #set xmin and xmax  

ax.axis([xmin,xmax,ymin,ymax]) 
ax.tick_params(top = 'off', right = 'off', bottom = 'off')
ax.spines['top'].set_position(('data',0))
ax.spines['right'].set_position(('data',0))
plt.gca().spines['bottom'].set_position(('data',0))
ax.spines['bottom'].set_visible(False)
plt.gca().axes.get_xaxis().set_visible(False)
 
plt.locator_params(axis='y', nbins=6)
plt.tick_params(labelsize= 0)
ax.ticklabel_format(useOffset=False)
if save_plots == True: 
    plt.savefig(savedir + exp_type + '_' + measurement + '_n-of-' + str(number) + filetype,bbox_inches='tight') #save figure 
if show_plots == True:
    plt.show() #display figure 
plt.close()