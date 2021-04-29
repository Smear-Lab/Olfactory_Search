'''
Figure 2B & Fig 2S1 -- Performance, trial duration, & path tortuosity across learning 

This code calculates and plots basic trial/session statistics for a chosen experimental condition across mice and plots it.
This should be used primarily for plotting learning across sessions in training steps of assay.  

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
# 

### --- USER INPUT --- ###

#EXPERIMENT INFO
datadir = "C:/Users/tfindley/Dropbox (University of Oregon)/Matt & Reese - Shared/Documents/Manuscripts/Reese_Manuscript_Search1/dryad/Findley files/"

measurement = 'mean_performance'; #represents trial/session statistic being calculated 
#possible measurements: mean_performance, trial_duration, tortuosity

exp_condition = ['100-0'] #this can be input as a single value or list (setting a list will generate multiple plots) 
#possible conditions: trainer1, trainer2, 100-0, 80-20, 60-40

#ANALYSIS PREFERENCES
error_report = False; #prints each time there are skipped files/trials/values
tracking_smoothing_window = 7; #rolling window for smoothing tracking data
min_trial_limit = 10; #minimum number of trials to accept a session
trial_time_limit = 10; #maximum time per trial
camera_framerate = 80; #framerate in Hz
tracking_jump_threshold = 40; #threshold for jumps in frame to frame nose position (in pixels) 

#PLOTTING PREFERENCES
#list of colors for each specific mouse (to map the same color to the same subject when running different analyses) 
color_list = [(0.36863,0.3098,0.63529),(0.2857,0.36983,0.68395),(0.23071,0.43015,0.71815),(0.2018,0.49075,0.73722),(0.19848,0.55214,0.73947),(0.23529,0.6196,0.71509),(0.3005,0.68667,0.67839),(0.37336,0.74331,0.65077),(0.44002,0.78249,0.64609),(0.51396,0.81261,0.64521),(0.59089,0.8385,0.64457),(0.66379,0.86399,0.64333),(0.73618,0.89455,0.63513),(0.81107,0.92774,0.61777),(0.87276,0.95314,0.60146),(0.90608,0.96074,0.59705),(0.92228,0.95945,0.62464),(0.93375,0.95657,0.67002),(0.94411,0.95243,0.70581),(0.95705,0.94619,0.70589),(0.97332,0.93181,0.66785),(0.98802,0.91033,0.61079),(0.99582,0.88406,0.55451),(0.99587,0.84909,0.50836),(0.99505,0.79785,0.46152),(0.99372,0.73769,0.41659),(0.99194,0.67629,0.37663),(0.98701,0.60891,0.33523),(0.97783,0.5355,0.29515),(0.96542,0.46486,0.268),(0.94982,0.40523,0.26428),(0.92366,0.35551,0.27786),(0.88838,0.30906,0.29646),(0.84788,0.25954,0.30893),(0.80334,0.20253,0.30792),(0.74965,0.14052,0.29882),(0.68788,0.074178,0.28237),(0.61961,0.0039216,0.25882)]

point_size = 10 #size of points
line_width = 1 #width of lines for individual mice
marker_type = 'o' #type of point
alpha_val = 0.8 #opacity value
average_line_width = 2.5 #line width for average across mice plots
point_edge_width = 0 #outline on points

#SAVING PREFERENCES
savedir = 'C:/Users/tfindley/Dropbox/analysis/general_stats/across-sessions/' #saving pathway
filetype = '.svg'
show_plot = True; #display plots while saving (for quality check step) 
save_plots = False #option to save figure or not 

#NOTE: x values for each experimental condition can be changed below. Lines: 74-77

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

#Check for saving directory -- if it does not exist, make it 
if (os.path.exists(savedir) == False):
    os.makedirs(savedir)
    print("Created Saving Folder") 
else: print ("Saving Folder Exists")

mpl.rcParams['savefig.pad_inches'] = 0 #initiate figure

#Find subjects in data directory 
subject_list = os.listdir(datadir) #walk through data directory for list of subjects 

#Run through each experimental condition listed (i.e. trainer 1, trainer 2, 100-0, etc.) 
for current_condition in exp_condition:
    
    #Plot settings -- can edit each of these for different number of sessions in each plot 
    fig = plt.figure(); ax = fig.add_subplot(111) #initiate figure  
    if current_condition == 'trainer1': xmin = 0; xmax = 10 #manually set x's (# of sessions plotted) so they are consistent across conditions
    if current_condition == 'trainer2': xmin = 0; xmax = 6
    if current_condition == '100-0': xmin = 0; xmax = 6
    if current_condition == '80-20': xmin = 0; xmax = 10
    
    mouse_list = np.zeros((1,1)); mouse_list.fill(np.nan); #create an array for mean measurement values for each mouse
    mouse_counter = 0 #count mice analyzed and plotted
    
    #Run through each mouse in previously generated subject list 
    for mouse_id in subject_list:
        working_dir = datadir + mouse_id + "/" + current_condition + "/" #working directory
        
        #If experimental directory does not exist, skip 
        if os.path.exists(working_dir) == False:
            if error_report == True:
                print mouse_id  + ': ' + current_condition + ': ' + 'No Experiment Directory -- skipping mouse'
            continue
        
        #Create list of sessions for individual mouse
        os.chdir(working_dir) #navigate to local directory
        session_list = [name for name in os.listdir(".") if os.path.isdir(name)] #find all session folders in working directory
        
        session_counter = 0; #count sessions analyzed and plotted
        pastsessiontime = 0; #initiate variable that counts time that passed between consecutive sessions 
        
        #Run through each session in list of all sessions for individual mouse 
        for session in session_list:

            measurement_values = [] #create list to store individual trial (or session) values of measurement being taken 
            session_dir = working_dir + str(session) + '/' #open session directory
            
            print mouse_id, current_condition, session #report working session to user
            
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

            '''
            Do not analyze sessions more than a week apart in 100-0 or 80-20!
            Mice are run across many weeks, but training occurs consecutively. Therefore, sessions more than a week apart occur post training and 
            should not be plotted with the training data. 
            '''
#             if current_condition == '100-0' or current_condition == '80-20':
#                 sessiontime = os.path.getmtime(bx_file) #acess date/time of session        
#                 if session == 1:
#                     pastsessiontime = sessiontime #set date/time for first session and move to next analysis step
#                     pass
#                 elif sessiontime - pastsessiontime > 604800: #if it's more than seven days since the last session, stop analysis of mouse
#                     if error_report == True:     
#                         print mouse_id, session, 'Too long between consecutive sessions -- ending analysis for current mouse' #report end of mouse analysis
#                         break
#                 else:
#                     pastsessiontime = sessiontime #update date/time of previous session
            
            #Pre-processing session information
            if len(trial_summ) < min_trial_limit: #if session is less than trial limit
                if error_report == True:
                    print mouse_id, current_condition, session, 'Session does not meet min. trial limit -- skipping session'
                continue
            
            #Pre-processing tracking data 
            if len(nx) < 1: #check x coordinates file for data 
                if error_report == True:
                    print mouse_id, current_condition, session, 'No data in nx file -- skipping session'
                continue
            #Smooth tracking data 
            nx = rolling_average(nx,tracking_smoothing_window); ny = rolling_average(ny,tracking_smoothing_window)
            
            #PERFORMANCE
            if measurement == 'mean_performance':
                measurement_values.append(np.sum(answer[concsetting != 3])/len(answer[concsetting != 3])*100) #include all data except odor omission trials
                if current_condition == 'trainer1' or current_condition == 'trainer2': #for initial trainers (where there is no in/correct answer)
                    measurement_values.append(len(trial_summ)) #count number of trials and record as performance
             
            if measurement != 'mean_performance':
                #Run through each trial of a session
                for current_trial in range (0,len(trial_summ)):
                    
                    if concsetting[current_trial] != 3: #remove all odor omission trials
                        start = tstart[current_trial]; end = tend[current_trial]; #mark start and end times of trial
                        
                        #if trial is too long, move to the next trial
                        if end - start > trial_time_limit:
                            if error_report == True:
                                print mouse_id, current_condition, session, current_trial, 'Trial too long -- skipping trial' #error report
                            continue
                        
                        #TRIAL DURATION
                        if measurement == 'trial_duration':
                            trialduration = end - start
                            measurement_values.append(trialduration)
                        
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
                                    continue #if mouse has crossed decision line -- exit loop
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
                                measurement_values.append(distance_travelled/shortest_distance)
            
            #take mean of trial values for a single measurement value per session
            sessionvalue = np.mean(measurement_values)
            mouse_list[mouse_counter,session_counter] = sessionvalue #add session value to mouse list
            
            #add a NaN column to mouse array for next session  
            if len(mouse_list[0,:]) <= session_counter+1:
                add_column = np.zeros((len(mouse_list[:,0]),1)); add_column.fill(np.nan)
                mouse_list = np.append(mouse_list, add_column, axis = 1)
            
            session_counter = session_counter + 1 #count analyzed and saved session 

        x_vals = np.arange(1,(len(mouse_list[mouse_counter,:])+1)) #set x axis for plotting session values of current mouse
        
        #Plot session values for current mouse in unique color 
        ax.plot(x_vals[0:xmax-1],mouse_list[mouse_counter,0:xmax-1], color = color_list[subject_list.index(mouse_id)],linewidth = line_width, marker = marker_type,markersize = point_size, markeredgewidth = point_edge_width, alpha = alpha_val)
        mouse_counter = mouse_counter + 1 #count analyzed and saved mouse
        
        #add a NaN row to mouse array for next mouse
        add_row = np.zeros((1,len(mouse_list[0,:]))); add_row.fill(np.nan)
        mouse_list = np.append(mouse_list, add_row, axis = 0)
        
    x_vals = np.arange(1,(len(mouse_list[0,:])+1)) #set x axis for plotting average across mice 
    mouse_list = (mouse_list[~np.isnan(mouse_list).all(axis=1)]) #remove any rows with nan values from the mouse_list
    
    #Plot the average across mice 
    for session_num in range(0,xmax - 1):
        plt.errorbar(x_vals[session_num],np.nanmean(mouse_list[:,session_num]), yerr = np.nanstd(mouse_list[:,session_num]), linewidth = 0, color = 'k',marker = 'o',markersize = point_size + 5,elinewidth = average_line_width, zorder = 32)
    #Set y values for plot depending on measurement and trainer 
    if measurement == 'mean_performance':
        ymin = 40; ymax = 100
        if current_condition == 'trainer1' or current_condition == 'trainer2':
            ymin = 0; ymax = 175
    if measurement == 'trial_duration':
        ymin = 0; ymax = 10
    if measurement == 'tortuosity':
        ymin = 1; ymax = 3
    if measurement == 'sniff_freq':
        ymin = 0; ymax = 350
    number = len(mouse_list[:,0]); print number #count and report number of mice used 
    
    #Plot settings 
    plt.xlim(xmin,xmax);plt.ylim(ymin,ymax)
    ax.tick_params(top = 'False', right = 'False', bottom = 'False')
    ax.spines['top'].set_position(('data',0))
    ax.spines['right'].set_position(('data',0))
    plt.gca().spines['bottom'].set_position(('data',0))
    ax.spines['bottom'].set_visible(False)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.tick_params(labelsize= 0)
    if save_plots == True: 
        plt.savefig(savedir + current_condition + '_' + measurement + '_n-of-' + str(number) + filetype,bbox_inches='tight') #save figure 
    if show_plot == True:
        plt.show() #display figure 
    plt.close() #close figure 

