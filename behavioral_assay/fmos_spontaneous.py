'''
FMOS Spontaneous - Freely Moving Olfactory Search - RECORDS SPONTANEOUS BEHAVIOR

Written: Teresa Findley, tfindley@uoregon.edu
Last Updated: 04.26.2021

--Records tracking data via OSC communication with custom code in Bonsai (open source computer vision software -- https://bonsai-rx.org/) 
--Records signal data through NI USB-6009 data acquisition board
--Controls solenoid and beambreak hardware through Arduino Mega2560 & Teensyduino 2.0
'''

#     [SET UP]     #

##IMPORTS
##libraries
import numpy as np, cv2, os
import time, math, random, datetime
from timeit import default_timer as timer
import OSC, threading, Queue
import nidaqmx, ctypes
import matplotlib.pyplot as plt
from nidaqmx.constants import AcquisitionType, Edge
from nidaqmx.stream_readers import AnalogMultiChannelReader
##local modules
from fmos_preferences_bonsai import *
import fmos_datamgt, fmos_tracking, fmos_serial

##INITIATE VARIABLES -- these are all state machine variables to be used throughout the session 
session_num = 1;

datapath,session_num = fmos_datamgt.CHK_directory(mouse_id,group_name,session_num) #update/create datapath
trialsummary_file = datapath + 'trialsummary.txt'; video_file = datapath + 'videolocation.txt'
notes_file = datapath + 'notes.txt'

ch0_file = datapath + ch0 + '.dat'; ch1_file = datapath + ch1 + '.dat' #NI signal files
ch2_file = datapath + ch2 + '.dat'; ch3_file = datapath + ch3 + '.dat'
nx_file = datapath + 'nosex.dat'; ny_file = datapath + 'nosey.dat' #bonsai tracking files
hx_file = datapath + 'headx.dat'; hy_file = datapath + 'heady.dat'
cx_file = datapath + 'comx.dat'; cy_file = datapath + 'comy.dat'
ts_file = datapath + 'timestamp.dat'

receive_address = ('localhost', 6666); trackingcoords = OSC.OSCServer(receive_address); #bonsai tracking variables
qnosex = Queue.LifoQueue(0); qnosey = Queue.LifoQueue(0); #online position storage
nosex = np.zeros((1,1)); nosey = np.zeros((1,1));
headx = np.zeros((1,1)); heady = np.zeros((1,1))
comx = np.zeros((1,1)); comy = np.zeros((1,1))
ts = np.zeros((1,1));
signaldata = np.zeros((channel_num,buffersize),dtype=np.float64) #NI data collection reading variables
reader = AnalogMultiChannelReader(ni_data.in_stream) 

#Session Summary

#Create/Open Data Files
ch0_handle = open(ch0_file,'ab'); ch1_handle = open(ch1_file,'ab'); ch2_handle = open(ch2_file,'ab'); ch3_handle = open(ch3_file,'ab');
nx_handle = open(nx_file,'ab'); ny_handle = open(ny_file,'ab'); hx_handle = open(hx_file,'ab') #nx nose coord, h head coord, c center of mass
hy_handle = open(hy_file,'ab'); cx_handle = open(cx_file,'ab'); cy_handle = open(cy_file,'ab')
ts_handle = open(ts_file,'ab') #time stamp

#Bonsai Start Up
trackingcoords.addDefaultHandlers() #add default handlers to the server
def msg_handler(addr, tags, coords, source):
    qnosex.put(coords[0]); qnosey.put(coords[1]); #online storage of nose position
    nosex[0,0] = coords[0]; nosey[0,0] = coords[1]
    headx[0,0] = coords[2]; heady[0,0] = coords[3]
    comx[0,0] = coords[4]; comy[0,0] = coords[5]
    ts[0,0] = timer()-session_start; 
    nosex.tofile(nx_handle); nosey.tofile(ny_handle)
    headx.tofile(hx_handle); heady.tofile(hy_handle)
    comx.tofile(cx_handle); comy.tofile(cy_handle)
    ts.tofile(ts_handle)
trackingcoords.addMsgHandler("/2python",msg_handler) #add msg handler function to server
bonsaitracking = threading.Thread( target = trackingcoords.serve_forever ) #put server in parallel thread
bonsaitracking.daemon = True

#NI Set Up
ni_data.ai_channels.add_ai_voltage_chan(channels) #add channels to server
ni_data.timing.cfg_samp_clk_timing(samplingrate, '',Edge.RISING,AcquisitionType.CONTINUOUS,uInt64(buffersize)) #instruct how to sample
def ni_handler(): #define background function to handle incoming NI data
    while True:
        reader.read_many_sample(signaldata,number_of_samples_per_channel= buffersize, timeout=10.0)
        signaldata[0,:].tofile(ch0_handle); signaldata[1,:].tofile(ch1_handle); 
        signaldata[2,:].tofile(ch2_handle); signaldata[3,:].tofile(ch3_handle);
nisignal = threading.Thread(target = ni_handler) #set handler function in background
nisignal.daemon = True

##INITIATE SESSION
print "Subject " + str(mouse_id) + ", Session " + str(session_num) #report session initiation
print "System Ready. Initiating Data Collection..."

bonsaitracking.start();
nose = [qnosex.get(),qnosey.get()];
session_start = timer() #session timer
ni_data.start(); nisignal.start(); #start data collection 
localtime = datetime.datetime.now(); #stamp for video locator

print "Session Started."


#     [MAIN CODE]     #

while True: 
#   [State *](occurs in all states)
    #Nosepoke & Timer

    if timer() - session_start >= session_length:
        break

    if show_active_stats == True: #real time trial statistics -- not necessary for spontaneous behavior, but still available for early session termination 
        frame = cv2.imread('D:/FMON_Project/data/statsbackground.jpeg')
        height, width, depth = frame.shape  #white background
        cv2.imshow('Select this window to manually end session',frame) 

    ##Manual Session Termination
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        reasonforend = "Manual Exit"
        break

#     [SHUT DOWN]     #

print "Session Ended." #report end of session
notepad = str(input("Please record notes here. Be precise and thorough. Write inside quotation marks with no space at the end.")) + '\n'
ch0_handle.close();ch1_handle.close();ch2_handle.close();ch3_handle.close();
nx_handle.close();ny_handle.close();hx_handle.close();hy_handle.close();cx_handle.close();cy_handle.close(); ts_handle.close()
print "Data Collection Ended" #report end of data collection

performance_report = '0'
fmos_datamgt.write_vidlocator(video_file,localtime)
fmos_datamgt.record_notes(notes_file,session_num,localtime,notepad, performance_report)

