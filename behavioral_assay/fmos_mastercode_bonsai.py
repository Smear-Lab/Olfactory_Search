'''
FMOS MASTERCODE - Freely Moving Olfactory Search Mastercode

Written: Teresa Findley, tmfindley15@gmail.com
Last Updated: 10.27.2020 (Dorian Yeh)

--Records tracking data via OSC communication with custom code in Bonsai (open source computer vision software -- https://bonsai-rx.org/) 
--Records signal data through NI USB-6009 data acquisition board
--Controls solenoid and beambreak hardware through Arduino Mega2560 & Teensyduino 2.0
'''

#     [SET UP]     #

##IMPORTS
##libraries
from numpy.random import choice
import numpy as np, cv2, os, sys
from timeit import default_timer as timer
import time, math, datetime, random
import OSC,threading, Queue
import nidaqmx, ctypes
import matplotlib.pyplot as plt
from nidaqmx.constants import AcquisitionType, Edge
from nidaqmx.stream_readers import AnalogMultiChannelReader

##local modules
from fmos_preferences_bonsai import *
import fmos_datamgt, fmos_tracking, fmos_serial

##INITIATE VARIABLES -- these are all state machine variables to be used throughout the session 

session_num = 1; trial_num = 1; state = 1; prep_odor = True; iti_delay = iti_correct; #trial information
correct0=0; correct1=0; correct2=0; correct3=0; correct4=0; total0=0; total1 = 0; total2=0; total3=0; total4=0; #real time performance statistics
correct0L=0; correct1L=0; correct2L=0; correct3L=0; correct4L=0; total0L=0; total1L=0; total2L=0; total3L=0; total4L=0;
correct0R=0; correct1R=0; correct2R=0; correct3R=0; correct4R=0; total0R=0; total1R=0; total2R=0; total3R=0; total4R=0;
total_trials = 0;  total_left = 0; total_right = 0; left_correct = 0; right_correct = 0 #real time performance statistics
total_correct = 0; fraction_correct = 0; fraction_left = 0; fraction_right = 0;

last_occupancy = 0; section_occupancy = 0; counter = 0; msg = 0 #real time Bonsai tracking & nose poke monitor 
odor_calibration = np.genfromtxt('D:/FMON_Project/data/olfactometercalibration.txt', delimiter = ',') #odor calibration array

datapath,session_num = fmos_datamgt.CHK_directory(mouse_id,group_name,session_num) #update/create datapath & initiate data files
trialsummary_file = datapath + 'trialsummary.txt'; video_file = datapath + 'videolocation.txt'; timestamp_file = datapath + 'timestamp.txt'
notes_file = datapath + 'notes.txt'

ch0_file = datapath + ch0 + '.dat'; ch1_file = datapath + ch1 + '.dat' #NI signal files
#ch2_file = datapath + ch2 + '.dat'; ch3_file = datapath + ch3 + '.dat'
nx_file = datapath + 'nosex.dat'; ny_file = datapath + 'nosey.dat' #bonsai tracking files
hx_file = datapath + 'headx.dat'; hy_file = datapath + 'heady.dat'
cx_file = datapath + 'comx.dat'; cy_file = datapath + 'comy.dat'
ts_file = datapath + 'timestamp.dat' #timestamp file

receive_address = ('localhost', 6666); trackingcoords = OSC.OSCServer(receive_address); #bonsai tracking variables
qnosex = Queue.LifoQueue(0); qnosey = Queue.LifoQueue(0); #real time position input 
nosex = np.zeros((1,1)); nosey = np.zeros((1,1));
headx = np.zeros((1,1)); heady = np.zeros((1,1))
comx = np.zeros((1,1)); comy = np.zeros((1,1))
ts = np.zeros((1,1));

signaldata = np.zeros((channel_num,buffersize),dtype=np.float64) #NI data collection reading variables
reader = AnalogMultiChannelReader(ni_data.in_stream) 

##START UP PROCEDURES
section,section_center=fmos_tracking.calc_partitions() #real time tracking: gridline deliniation (depends on rig size) 
triallist,odorconditionlist = fmos_datamgt.randomize_trials(random_groupsize,total_groupsize) #randomize trials
fmos_serial.close_all_valves() #turn off all hardware

#Create/Open Data Files
ch0_handle = open(ch0_file,'ab'); ch1_handle = open(ch1_file,'ab'); 
#ch2_handle = open(ch2_file,'ab'); ch3_handle = open(ch3_file,'ab');
nx_handle = open(nx_file,'ab'); ny_handle = open(ny_file,'ab'); hx_handle = open(hx_file,'ab')
hy_handle = open(hy_file,'ab'); cx_handle = open(cx_file,'ab'); cy_handle = open(cy_file,'ab')
ts_handle = open(ts_file,'ab')

#Bonsai Start Up
trackingcoords.addDefaultHandlers() #add default handlers to the server
def msg_handler(addr, tags, coords, source):
    qnosex.put(coords[0]); qnosey.put(coords[1]); #real time storage of nose position
    nosex[0,0] = coords[0]; nosey[0,0] = coords[1]
    headx[0,0] = coords[2]; heady[0,0] = coords[3]
    comx[0,0] = coords[4]; comy[0,0] = coords[5]
    ts[0,0] = timer()-session_start; 
    nosex.tofile(nx_handle); nosey.tofile(ny_handle) #save nose, head, and body coordinates in real time 
    headx.tofile(hx_handle); heady.tofile(hy_handle)
    comx.tofile(cx_handle); comy.tofile(cy_handle)
    ts.tofile(ts_handle)
trackingcoords.addMsgHandler("/2python",msg_handler) #add msg handler function to server for between program communication 
bonsaitracking = threading.Thread( target = trackingcoords.serve_forever ) #put tracking in continuous background thread
bonsaitracking.daemon = True

#NI Set Up
ni_data.ai_channels.add_ai_voltage_chan(channels) #add channels to server
ni_data.timing.cfg_samp_clk_timing(samplingrate, '',Edge.RISING,AcquisitionType.CONTINUOUS,uInt64(buffersize)) #instruct how to sample
def ni_handler(): #define background function to handle incoming NI data
    while True:
        reader.read_many_sample(signaldata,number_of_samples_per_channel= buffersize, timeout=10.0)
        signaldata[0,:].tofile(ch0_handle); signaldata[1,:].tofile(ch1_handle); 
        #signaldata[2,:].tofile(ch2_handle); signaldata[3,:].tofile(ch3_handle);
nisignal = threading.Thread(target = ni_handler) #set handler function in background
nisignal.daemon = True

##INITIATE SESSION
print ("Subject " + str(mouse_id) + ", Session " + str(session_num)) #report session initiation
print ("System Ready. Initiating Data Collection...")
print ("Did you remember to turn on the NITROGEN??") #reminder for users

bonsaitracking.start(); #initiate waiting for Bonsai input 
nose = [qnosex.get(),qnosey.get()]; #ask for input from Bonsai
#**********PROGRAM WILL NOT CONTINUE UNTIL IT RECEIVES INPUT...START BONSAI PROGRAM HERE**********#
session_start = timer() #session timer
ni_data.start(); nisignal.start(); #start NIDAQ sniff data collection 
localtime = datetime.datetime.now(); #timestamp for locating videos saved locally through Bonsai 

print ("Session Started.")


#     [MAIN CODE]     #
while True: 
#   [State *](occurs across all states in state machine)
    
    #Nosepoke & Timer
    while ard.inWaiting() > 0: #check nosepoke status
        msg = fmos_serial.nose_poke_status(msg)   
        
    if timer() - session_start >= session_length: #end session at predetermined length 
        fmos_serial.close_all_valves()
        reasonforend = "Auto Session End"
        break
    
    #Realtime Tracking 
    nose = [qnosex.get(),qnosey.get()]; #check nose position
    section_occupancy = fmos_tracking.detect_mouse_partitions(nose,section_center, section_occupancy) #section occupancy
    
    if show_active_stats == True: #real time trial statistics
        frame = cv2.imread('D:/FMON_Project/data/statsbackground.jpeg')
        height, width, depth = frame.shape  #white background
        fraction_correct = "T:     "+str(correct0)+"/"+str(total0)+".    "+str(correct4)+"/"+str(total4)+".     "+str(correct1)+"/"+str(total1)+".    "+str(correct2)+"/"+str(total2)+".     "+str(correct3)+"/"+str(total3)+".    "  #session stats
        fraction_left = "L:     "+str(correct0L)+"/"+str(total0L)+".    "+str(correct4L)+"/"+str(total4L)+".     "+str(correct1L)+"/"+str(total1L)+".    "+str(correct2L)+"/"+str(total2L)+".     "+str(correct3L)+"/"+str(total3L)+"."
        fraction_right = "R:     "+str(correct0R)+"/"+str(total0R)+".    "+str(correct4R)+"/"+str(total4R)+".     "+str(correct1R)+"/"+str(total1R)+".    "+str(correct2R)+"/"+str(total2R)+".     "+str(correct3R)+"/"+str(total3R)+"."
        #Stats Display
        if group_name == 'abs-conc':
            cv2.putText(frame,'xxxx    80-20(1%) 80-20(0.1%)  CONTROL', (130,(height/2)-40), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0))
        if group_name == 'thresholding':
            cv2.putText(frame,'xxxx    90-30   30-10  CONTROL', (130,(height/2)-40), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0))
        if group_name == 'non-spatial':
            cv2.putText(frame,'xxxx    Exp.   xxxx  CONTROL', (130,(height/2)-40), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0))
            fraction_left = "O(L):   "+str(correct0L)+"/"+str(total0L)+".     "+str(correct1L)+"/"+str(total1L)+".     "+str(correct2L)+"/"+str(total2L)+".     "+str(correct3L)+"/"+str(total3L)+". "
            fraction_right = "MS(R):   "+str(correct0R)+"/"+str(total0R)+".     "+str(correct1R)+"/"+str(total1R)+".     "+str(correct2R)+"/"+str(total2R)+".     "+str(correct3R)+"/"+str(total3R)+". "
        if group_name != 'abs-conc' and group_name != 'non-spatial' and group_name != 'thresholding':
            cv2.putText(frame,'100-0 | 90-10 | 80-20 | 60-40 | CONTROL', (130,(height/2)-40), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0))
        if group_name == 'mineral-oil':
            cv2.putText(frame,'M6.80-20   M6.50-50   M7.80-20   M7.50-50', (130,(height/2)-40), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0))
        cv2.putText(frame,fraction_correct, (80,(height/2)-20), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0))
        cv2.putText(frame,fraction_left,(80,(height/2)),cv2.FONT_HERSHEY_PLAIN,1,(0,0,0))
        cv2.putText(frame,fraction_right,(80,(height/2)+20),cv2.FONT_HERSHEY_PLAIN,1,(0,0,0))
        cv2.imshow('Session Statistics',frame) 

    ##Manual Session Termination -- press 'q' to end session manually 
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        fmos_serial.close_all_valves()
        reasonforend = "Manual Exit"
        break


#   [State 1] TRIAL INITIATION   
    if state == 1: 
        
        #Odor Preparation    
        if prep_odor == True:
            active_valve = triallist[trial_num-1] #side of odor delivery
            concentration_setting = odorconditionlist[trial_num-1] #concentration difference of odor delivery 
            #Update Trial Values & MFC settings
            low_valve, correctpoke,nameoftrialtype,correctindex,incorrectindex = fmos_datamgt.trial_values(active_valve)
            HairR,LairR,HairL,LairL,Hn2R,Ln2R,Hn2L,Ln2L,activevial,lowvial = fmos_serial.MFC_settings(concentration_setting,odor_calibration,active_valve)
            print ("Upcoming Trial: " + nameoftrialtype + ", " + str(concentration_setting)) #report upcoming trial
            #turn on MFCs and Vials
            if group_name != 'non-spatial':
                if active_valve == 1:
                    tnsy.write("MFC " + str(active_valve) + " " + str(MFC_air) + " " + str(HairR) + "\r")
                    tnsy.write("MFC " + str(active_valve) + " " + str(MFC_n2) + " " + str(Hn2R) + "\r")
                    tnsy.write("MFC " + str(low_valve) + " " + str(MFC_air) + " " + str(LairL) + "\r")
                    tnsy.write("MFC " + str(low_valve) + " " + str(MFC_n2) + " " + str(Ln2L) + "\r")
                if active_valve == 2: 
                    tnsy.write("MFC " + str(active_valve) + " " + str(MFC_air) + " " + str(HairL) + "\r")
                    tnsy.write("MFC " + str(active_valve) + " " + str(MFC_n2) + " " + str(Hn2L) + "\r")
                    tnsy.write("MFC " + str(low_valve) + " " + str(MFC_air) + " " + str(LairR) + "\r")
                    tnsy.write("MFC " + str(low_valve) + " " + str(MFC_n2) + " " + str(Ln2R) + "\r")
                tnsy.write("vialOn " + str(active_valve) + " " + str(activevial) + "\r")
                tnsy.write("vialOn " + str(low_valve) + " " + str(lowvial) + "\r")

            if group_name == 'non-spatial':
                if non_spatial_condition == 'odor_concentration': 
                    if active_valve == 1:
                        tnsy.write("MFC " + str(active_valve) + " " + str(MFC_air) + " " + str(HairR) + "\r")
                        tnsy.write("MFC " + str(active_valve) + " " + str(MFC_n2) + " " + str(Hn2R) + "\r")
                        tnsy.write("MFC " + str(low_valve) + " " + str(MFC_air) + " " + str(HairL) + "\r")
                        tnsy.write("MFC " + str(low_valve) + " " + str(MFC_n2) + " " + str(Hn2L) + "\r")
                    if active_valve == 2: 
                        tnsy.write("MFC " + str(active_valve) + " " + str(MFC_air) + " " + str(LairL) + "\r")
                        tnsy.write("MFC " + str(active_valve) + " " + str(MFC_n2) + " " + str(Ln2L) + "\r")
                        tnsy.write("MFC " + str(low_valve) + " " + str(MFC_air) + " " + str(LairR) + "\r")
                        tnsy.write("MFC " + str(low_valve) + " " + str(MFC_n2) + " " + str(Ln2R) + "\r")
                    tnsy.write("vialOn " + str(active_valve) + " " + str(odor_vial) + "\r")
                    tnsy.write("vialOn " + str(low_valve) + " " + str(odor_vial) + "\r")
                if non_spatial_condition == 'odor_identity': 
                    tnsy.write("MFC " + str(active_valve) + " " + str(MFC_air) + " " + str(HairR) + "\r")
                    tnsy.write("MFC " + str(active_valve) + " " + str(MFC_n2) + " " + str(Hn2R) + "\r")
                    tnsy.write("MFC " + str(low_valve) + " " + str(MFC_air) + " " + str(HairL) + "\r")
                    tnsy.write("MFC " + str(low_valve) + " " + str(MFC_n2) + " " + str(Hn2L) + "\r")
                    if active_valve == 1:
                        tnsy.write("vialOn " + str(active_valve) + " " + str(odor_vial) + "\r")
                        tnsy.write("vialOn " + str(low_valve) + " " + str(odor_vial) + "\r")
                    if active_valve == 2:
                        tnsy.write("vialOn " + str(active_valve) + " " + str(odor_vial2) + "\r")
                        tnsy.write("vialOn " + str(low_valve) + " " + str(odor_vial2) + "\r")                        

            iti_timeout_start = math.floor(timer()) #start vial timer
            prep_odor = False #odor has been decided
        
        #Trial Initiation
        if (math.floor(timer()) >= math.floor(iti_timeout_start + iti_delay)): #vial mixing timer
            if msg == 3:
                tstart = timer() - session_start; #timestamp trial start (in ms)
                tnsy.write("valve " + str(low_valve) + " 1 on\r") #turn on FVs
                tnsy.write("valve " + str(active_valve) + " 1 on\r")
                state = 2 #update trial variables
                print (("Trial " + str(trial_num) + " Activated: " + nameoftrialtype)) #report trial start


#   [State 2] TRIAL DECISION
    if state == 2:
        #Frame Count of Section Occupancy
        if (section_occupancy == last_occupancy):
            if (section_occupancy < 2):
                counter = counter + 1 
            else: counter = 0; last_occupancy = section_occupancy 
        else: counter = 0; last_occupancy = section_occupancy 
        
        #Decision Status
        if (counter == count_requirement): 
            if (section_occupancy == correctindex):
                response = 1; answer = "Correct"
            elif (section_occupancy == incorrectindex):
                response = 0; answer = "Incorrect"
            print("Response registered: " + answer) #report response
            tnsy.write("valve " + str(active_valve) + " 1 off\r") #turn off final valves
#            tnsy.write("valve " + str(low_valve) + " 1 off\r") 
            state = 3; counter = 0;  #update trial statistics

#   [State 3] REWARD DELIVERY
    if state == 3:    
        #Correct Responses
        if response == 1:            
            if msg == correctpoke: 
                if active_valve == 1: #Increment Active Statistics
                    if concentration_setting == 0:
                        total0 = total0 + 1; total0R = total0R + 1; correct0 = correct0 + 1; correct0R = correct0R + 1
                    if concentration_setting == 1: 
                        total1 = total1 + 1; total1R = total1R + 1; correct1 = correct1 + 1; correct1R = correct1R + 1
                    if concentration_setting == 2: 
                        total2 = total2 + 1; total2R = total2R + 1; correct2 = correct2 + 1; correct2R = correct2R + 1
                    if concentration_setting == 3: 
                        total3 = total3 + 1; total3R = total3R + 1; correct3 = correct3 + 1; correct3R = correct3R + 1
                    #90-10 R correct-------------
                    if concentration_setting == 4: 
                        total4 = total4 + 1; total4R = total4R + 1; correct4 = correct4 + 1; correct4R = correct4R + 1
                    if concentration_setting == 5: 
                        total4 = total4 + 1; total4R = total4R + 1; correct4 = correct4 + 1; correct4R = correct4R + 1
                    #----------------------------
                if active_valve == 2:
                    if concentration_setting == 0:
                        total0 = total0 + 1; total0L = total0L + 1; correct0 = correct0 + 1; correct0L = correct0L + 1
                    if concentration_setting == 1: 
                        total1 = total1 + 1; total1L = total1L + 1; correct1 = correct1 + 1; correct1L = correct1L + 1
                    if concentration_setting == 2: 
                        total2 = total2 + 1; total2L = total2L + 1; correct2 = correct2 + 1; correct2L = correct2L + 1
                    if concentration_setting == 3: 
                        total3 = total3 + 1; total3L = total3L + 1; correct3 = correct3 + 1; correct3L = correct3L + 1
                    #90-10 L correct--------------------------
                    if concentration_setting == 4: 
                        total4 = total4 + 1; total4L = total4L + 1; correct4 = correct4 + 1; correct4L = correct4L + 1
                    if concentration_setting == 5: 
                        total4 = total4 + 1; total4L = total4L + 1; correct4 = correct4 + 1; correct4L = correct4L + 1
                    #-------------------------------
                fmos_serial.deliver_reward(msg) #deliver reward
                print("Reward Delivered.") #report reward delivery
                tend = timer() - session_start #timestamp trial end & record trial summary info
                fmos_datamgt.write_trialsummary(trialsummary_file,trial_num,concentration_setting, active_valve,response,tstart,tend)
                state = 1; prep_odor = True; iti_delay = iti_correct;trial_num = trial_num + 1;  #update trial variables 
        
        #Incorrect Responses
        else: 
            if msg > 0:
                if active_valve == 1: #Increment Active Statistics
                    if concentration_setting == 0:
                        total0 = total0 + 1; total0R = total0R + 1; 
                    if concentration_setting == 1: 
                        total1 = total1 + 1; total1R = total1R + 1; 
                    if concentration_setting == 2: 
                        total2 = total2 + 1; total2R = total2R + 1;
                    if concentration_setting == 3: 
                        total3 = total3 + 1; total3R = total3R + 1;
                    #90-10 R wrong-----------------
                    if concentration_setting == 4: 
                        total4 = total4 + 1; total4R = total4R + 1;
                    if concentration_setting == 5:
                         total4 = total4 + 1; total4R = total4R + 1;
                    #----------------------------
                        
                if active_valve == 2:
                    if concentration_setting == 0:
                        total0 = total0 + 1; total0L = total0L + 1;
                    if concentration_setting == 1: 
                        total1 = total1 + 1; total1L = total1L + 1;
                    if concentration_setting == 2: 
                        total2 = total2 + 1; total2L = total2L + 1;
                    if concentration_setting == 3: 
                        total3 = total3 + 1; total3L = total3L + 1;
                    #90-10 L wrong-------------
                    if concentration_setting == 4: 
                        total4 = total4 + 1; total4L = total4L + 1;
                    if concentration_setting == 5:
                         total4 = total4 + 1; total4L = total4L + 1;
                    #---------------------------
                print("No Reward Delivered.") #report no reward
                tend = timer() - session_start #timestamp trial end & record trial summary info 
                fmos_datamgt.write_trialsummary(trialsummary_file,trial_num,concentration_setting,active_valve,response,tstart,tend)
                state = 1; prep_odor = True; trial_num = trial_num + 1;  #update trial variables
                if concentration_setting == 3:
                    iti_delay = iti_correct
                else: iti_delay = iti_incorrect

#     [SHUT DOWN]     #

tnsy.write("vialOff " + str(right_valve) + " " + str(lowvial) + "\r")
tnsy.write("vialOff " + str(left_valve) + " " + str(lowvial) + "\r")

notepad = str(input("Please record notes here. Be precise and thorough. Write inside quotation marks with no space at the end.")) + '\n'

#Close All Data Files
ch0_handle.close();ch1_handle.close();
#ch2_handle.close();ch3_handle.close();
nx_handle.close();ny_handle.close();hx_handle.close();hy_handle.close();cx_handle.close();cy_handle.close(); ts_handle.close()
print ("Session Ended.") #report end of session
print ("Data Collection Ended") #report end of data collection

##EXIT PROGRAM
fmos_serial.close_all_valves(); cv2.destroyAllWindows(); ard.close(); tnsy.close()

fraction_correct = "T: "+str(correct0)+"/"+str(total0)+".  "+str(correct4)+"/"+str(total4)+".  "+str(correct1)+"/"+str(total1)+".  "+str(correct2)+"/"+str(total2)+".  "+str(correct3)+"/"+str(total3)+"."  #session stats
fraction_left = "L: "+str(correct0L)+"/"+str(total0L)+".  "+str(correct4L)+"/"+str(total4L)+".  "+str(correct1L)+"/"+str(total1L)+".  "+str(correct2L)+"/"+str(total2L)+".  "+str(correct3L)+"/"+str(total3L)+"."
fraction_right = "R: "+str(correct0R)+"/"+str(total0R)+".  "+str(correct4R)+"/"+str(total4R)+".  "+str(correct1R)+"/"+str(total1R)+".  "+str(correct2R)+"/"+str(total2R)+".  "+str(correct3R)+"/"+str(total3R)+"."



if group_name == 'abs-conc' or group_name == 'non-spatial':
    if group_name == 'abs-conc':
        print ('   xxxx 80-20(1%) 80-20(0.1%) CONTROL')
    elif group_name == 'non-spatial':
        print ('   xxxx M.S.      Octanol     CONTROL')
else: print ('   100-0 90-10 80-20 60-40 CONTROL')
print (fraction_correct)
print (fraction_left)
print (fraction_right)

#Write Video Locator & Timestamp
fmos_datamgt.write_vidlocator(video_file,localtime)

performance_report = fraction_correct + '\n' + fraction_left + '\n' + fraction_right 
fmos_datamgt.record_notes(notes_file,session_num,localtime,notepad, performance_report)
