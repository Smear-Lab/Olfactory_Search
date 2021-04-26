'''
Olfactory Search Module 
Data Management (functions for working with data & updating trial values) 

Written By: Teresa Findley (tmfindley15@gmail.com)
Last Updated: 10.7.2020, Dorian Yeh
'''

#     [SET UP]     #

##IMPORTS
##libraries
import os, cv2, datetime,random
import numpy as np
##local modules
#from AUTO_preferences import * #AUTO is used for running automated 2 second trials without a mouse 
from fmos_preferences_bonsai import *

#     [FUNCTIONS]     #

# Create path for saving data
def CHK_directory(mouse_id,group_name,session_num):
    while True:
        datapath = base_dir + "Mouse_" + mouse_id + "/" + group_name + "/" + str(session_num) + "/"
        if (os.path.exists(datapath) == False):
            os.makedirs(datapath)
            break
        else: session_num = session_num + 1
    return datapath, session_num

#Record name of corresponding video that will be saved through Bonsai (saved as timestamp) 
def write_vidlocator(video_file,localtime):
    sess = open(video_file,'a')
    sess.write('rawvideo' + localtime.strftime('%Y-%m-%dT%H_%M_%S') + '.avi') #video location
    sess.close()

#Record notes file with session information & user input 
def record_notes(notes_file,session_num,localtime,notepad, performance_report):
    notes = open(notes_file,'a')
    note = 'Experimenter: ' + experimenter + '\n'
    notes.write(note)
    note = 'Trainer: ' + trainer + '\n'
    notes.write(note)
    note = 'Mouse sex: ' + sex + '\n'
    notes.write(note)
    note = "Date: " + localtime.strftime('%Y-%m-%d, %H:%M:%S') + '\n'
    notes.write(note)
    note = 'Mouse ID: ' + mouse_id + '\n'
    notes.write(note)
    note = 'Initial weight of mouse: ' + initial_weight + '\n'
    notes.write(note)
    note = 'Forced to one side (0=neither, 1=right, 2=left): ' + str(side_bias) + '\n' 
    notes.write(note)
    note = "Session: " + str(session_num) + '\n\n'
    notes.write(note)
    note = "Sniffing: " + sniffing + '\n'
    notes.write(note)
    note = "Occluded Nostril: " + occluded_nostril + '\n'
    notes.write(note)
    note = "Odor Type: " + odortype + '\n'
    notes.write(note)
    note = "Odor Percentage: " + percentinvial + '\n\n'
    notes.write(note) 
    note = "WATER CALIBRATION\nLeft Port: " + wcL + '\n'
    notes.write(note) 
    note = "Right Port: " + wcR + '\n'
    notes.write(note) 
    note = "Initiation Port: " + wcIP + '\n\n'
    notes.write(note)   
    note = "PERFORMANCE" + '\n' + performance_report + '\n\n'
    notes.write(note)
    note = "EXPERIMENTERS NOTES: " + notepad + '\n'
    notes.write(note)
    if group_name == 'doi':
        note = "Drug dose: " + drug_dose
        notes.write(note)
    notes.close()

# Record summary of results for each trial
def write_trialsummary(trialsummary_file,trial_num,concentration_setting,active_valve,response,tstart,tend):
    results = [np.nan]*6
    results[0] = str(trial_num)
    results[1] = str(concentration_setting)
    results[2] = active_valve
    results[3] = response
    results[4] = tstart
    results[5] = tend
    ts_handle = file(trialsummary_file,'a')
    np.savetxt(ts_handle,[results],delimiter=',',fmt="%s")
    ts_handle.close

# Randomize left/right trials & concentration setting (100-0,80-20,60-40,etc.) 
def randomize_trials(random_groupsize,total_groupsize):
    randylist = [] #list that will contain trial order
    randylist2 = [] #list containing odor condition order
    for i in range (0,(total_groupsize/random_groupsize)):
        if side_bias == 0: #trials on both sides
            randygen  = [1]*int(random_groupsize/2)+[2]*int(random_groupsize/2) #1 = right, 2 = left
        if side_bias == 1: #trials only on right side
            randygen  = [1]*int(random_groupsize/2)+[1]*int(random_groupsize/2) #1 = right
        if side_bias == 2: #trials only on left side 
            randygen  = [2]*int(random_groupsize/2)+[2]*int(random_groupsize/2) #2 = left
        #0 = 100-0, 1 = 80-20, 2 = 60-40, 3 = controls(10% of trials)
        if group_name == 'interleaved':
            randygen2 = [0]*int(random_groupsize*.3)+[1]*int(random_groupsize*.3)+[2]*int(random_groupsize*.3)+[3]*int(random_groupsize*.1)
        if group_name == 'abs-conc' or group_name == 'thresholding':
            randygen2 = [1]*int(random_groupsize*.4)+[2]*int(random_groupsize*.4)+[3]*int(random_groupsize*.2)
        if group_name == '100-0' or group_name == 'trainer2' or group_name == 'trainer2_non-spatial':
            randygen2 = [0]*int(random_groupsize*.9)+[3]*int(random_groupsize*.1)
        #-----------------------
        if group_name == '90-10':
            randygen2 = [4]*int(random_groupsize*.9)+[3]*int(random_groupsize*.1)
        if group_name == '90-10_alt':
            randygen2 = [5]*int(random_groupsize*.9)+[3]*int(random_groupsize*.1)
        if group_name == '90-10_interleaved':
            randygen2 = [4]*int(random_groupsize*.45)+[5]*int(random_groupsize*.45)+[3]*int(random_groupsize*.1)
        #-----------------------    
        if group_name == '80-20' or group_name == 'non-spatial' or group_name == 'nostril-occlusion':
            randygen2 = [1]*int(random_groupsize*.9)+[3]*int(random_groupsize*.1)
        if group_name == '60-40':
            randygen2 = [2]*int(random_groupsize*.9)+[3]*int(random_groupsize*.1)
        if group_name == 'control':
            randygen2 = [3]*int(random_groupsize*.9)+[3]*int(random_groupsize*.1)
        if group_name == 'mineral-oil':
            randygen2 = [0]*int(random_groupsize*.25)+[1]*int(random_groupsize*.25)+[2]*int(random_groupsize*.25)+[3]*int(random_groupsize*.25)
        random.shuffle(randygen); randylist.extend(randygen);
        random.shuffle(randygen2); randylist2.extend(randygen2);
    return randylist, randylist2

# Update valve commands and index information for each trial depending on active valve (side releasing higher concentration of odor) 
def trial_values(active_valve):
    if (active_valve) == 2:
        low_valve = 1; correctpoke = 2; nameoftrialtype = 'Left'; correctindex = 1; incorrectindex = 0
    if (active_valve) == 1:
        low_valve = 2; correctpoke = 1; nameoftrialtype = 'Right'; correctindex = 0; incorrectindex = 1
    return low_valve, correctpoke, nameoftrialtype, correctindex, incorrectindex

# Increment Reporting Statistics -- real time update on mouse performance 
def increment_stats(active_valve,total_trials,total_left,total_right,
                    total_correct,left_correct,right_correct,response):
    total_trials = total_trials + 1
    if active_valve == left_valve: total_left = total_left + 1
    else: total_right = total_right + 1
    if response == True:
        total_correct = total_correct + 1
        if active_valve == left_valve:
            left_correct = left_correct + 1 
        else: right_correct = right_correct + 1
    return(total_trials,total_left,total_right,total_correct,left_correct,right_correct)

