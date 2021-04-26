'''
Olfactory Search Module 
Serial Communication (Teensy & Arduino Boards)

Written By: Teresa Findley (tmfindley15@gmail.com)
Last Updated: 04.26.2021, Teresa Findley
'''

#     [SET UP]     #

##IMPORTS
##local modules
import random
#from AUTO_preferences import * #AUTO is used for running automated 2 second trials without a mouse 
from fmos_preferences_bonsai import *

#     [FUNCTIONS]     #

# Use odor calibration document to pull MFC settings for each trial
def MFC_settings(concentration_setting,odor_calibration,active_valve):
    if group_name == 'non-spatial': 
        HairR = odor_calibration[1,0]; Hn2R = odor_calibration[1,1]
        HairL = odor_calibration[1,2]; Hn2L = odor_calibration[1,3]
        LairR = odor_calibration[1,4]; Ln2R = odor_calibration[1,5]
        LairL = odor_calibration[1,6]; Ln2L = odor_calibration[1,7]
        activevial = odor_vial; lowvial = odor_vial           
    elif group_name == 'mineral-oil':
        #vial6.80-20
        if concentration_setting == 0:
            HairR = odor_calibration[1,0]; Hn2R = odor_calibration[1,1]
            HairL = odor_calibration[1,2]; Hn2L = odor_calibration[1,3]
            LairR = odor_calibration[1,4]; Ln2R = odor_calibration[1,5]
            LairL = odor_calibration[1,6]; Ln2L = odor_calibration[1,7]
            activevial = 6; lowvial = 6
        #vial6.50-50
        if concentration_setting == 1:
            HairR = odor_calibration[3,0]; Hn2R = odor_calibration[3,1]
            HairL = odor_calibration[3,2]; Hn2L = odor_calibration[3,3]
            LairR = odor_calibration[3,4]; Ln2R = odor_calibration[3,5]
            LairL = odor_calibration[3,6]; Ln2L = odor_calibration[3,7]
            activevial = 6; lowvial = 6
        #vial7.80-20
        if concentration_setting == 2:
            HairR = odor_calibration[1,0]; Hn2R = odor_calibration[1,1]
            HairL = odor_calibration[1,2]; Hn2L = odor_calibration[1,3]
            LairR = odor_calibration[1,4]; Ln2R = odor_calibration[1,5]
            LairL = odor_calibration[1,6]; Ln2L = odor_calibration[1,7]
            activevial = 7; lowvial = 7
        #vial7.50-50
        if concentration_setting == 3:
            HairR = odor_calibration[3,0]; Hn2R = odor_calibration[3,1]
            HairL = odor_calibration[3,2]; Hn2L = odor_calibration[3,3]
            LairR = odor_calibration[3,4]; Ln2R = odor_calibration[3,5]
            LairL = odor_calibration[3,6]; Ln2L = odor_calibration[3,7]
            activevial = 7; lowvial = 7        
    else:
        #100-0 Condition
        if concentration_setting == 0:
            HairR = odor_calibration[0,0]; Hn2R = odor_calibration[0,1]
            HairL = odor_calibration[0,2]; Hn2L = odor_calibration[0,3]
            LairR = odor_calibration[0,0]; Ln2R = odor_calibration[0,1]
            LairL = odor_calibration[0,2]; Ln2L = odor_calibration[0,3]
            ### CHANGE THIS ONE??? - Dorian 2.12
            activevial = odor_vial; lowvial = blank_vial
        #80-20 Condition
        if concentration_setting == 1:
            HairR = odor_calibration[1,0]; Hn2R = odor_calibration[1,1]
            HairL = odor_calibration[1,2]; Hn2L = odor_calibration[1,3]
            LairR = odor_calibration[1,4]; Ln2R = odor_calibration[1,5]
            LairL = odor_calibration[1,6]; Ln2L = odor_calibration[1,7]
            activevial = odor_vial; lowvial = odor_vial
            if group_name == 'thresholding':
                HairR = odor_calibration[4,0]; Hn2R = odor_calibration[4,1]
                HairL = odor_calibration[4,2]; Hn2L = odor_calibration[4,3]
                LairR = odor_calibration[4,4]; Ln2R = odor_calibration[4,5]
                LairL = odor_calibration[4,6]; Ln2L = odor_calibration[4,7]
                activevial = odor_vial; lowvial = odor_vial
        #60-40 Condition
        if concentration_setting == 2:
            if group_name == 'abs-conc':
                HairR = odor_calibration[1,0]; Hn2R = odor_calibration[1,1]
                HairL = odor_calibration[1,2]; Hn2L = odor_calibration[1,3]
                LairR = odor_calibration[1,4]; Ln2R = odor_calibration[1,5]
                LairL = odor_calibration[1,6]; Ln2L = odor_calibration[1,7]
                activevial = odor_vial2; lowvial = odor_vial2
            elif group_name == 'thresholding':
                HairR = odor_calibration[5,0]; Hn2R = odor_calibration[5,1]
                HairL = odor_calibration[5,2]; Hn2L = odor_calibration[5,3]
                LairR = odor_calibration[5,4]; Ln2R = odor_calibration[5,5]
                LairL = odor_calibration[5,6]; Ln2L = odor_calibration[5,7]
                activevial = odor_vial; lowvial = odor_vial
            else:             
                HairR = odor_calibration[2,0]; Hn2R = odor_calibration[2,1]
                HairL = odor_calibration[2,2]; Hn2L = odor_calibration[2,3]
                LairR = odor_calibration[2,4]; Ln2R = odor_calibration[2,5]
                LairL = odor_calibration[2,6]; Ln2L = odor_calibration[2,7]
                activevial = odor_vial; lowvial = odor_vial
                
        #90-10 condition
        if concentration_setting == 4:
            HairR = odor_calibration[4,0]; Hn2R = odor_calibration[4,1]
            HairL = odor_calibration[4,2]; Hn2L = odor_calibration[4,3]
            LairR = odor_calibration[4,4]; Ln2R = odor_calibration[4,5]
            LairL = odor_calibration[4,6]; Ln2L = odor_calibration[4,7]
            activevial = odor_vial; lowvial = odor_vial
        #90-10_alt condition (interleaved odors with 90-10)
        if concentration_setting == 5:
            HairR = odor_calibration[5,0]; Hn2R = odor_calibration[5,1]
            HairL = odor_calibration[5,2]; Hn2L = odor_calibration[5,3]
            LairR = odor_calibration[5,4]; Ln2R = odor_calibration[5,5]
            LairL = odor_calibration[5,6]; Ln2L = odor_calibration[5,7]
            activevial = alternate_vial; lowvial = odor_vial
           
        #Blank Condition
        if concentration_setting == 3:
            HairR = odor_calibration[1,0]; Hn2R = odor_calibration[1,1]
            HairL = odor_calibration[1,2]; Hn2L = odor_calibration[1,3]
            LairR = odor_calibration[1,4]; Ln2R = odor_calibration[1,5]
            LairL = odor_calibration[1,6]; Ln2L = odor_calibration[1,7]
            activevial = blank_vial; lowvial = blank_vial
    return HairR,LairR,HairL,LairL,Hn2R,Ln2R,Hn2L,Ln2L, activevial, lowvial

# Check nose poke status for all three nose pokes 
def nose_poke_status(msg):
    input = ard.readline().strip()
    if len(input) < 2:
        try:
            msg = int(input)
        except ValueError:
            pass
    return msg

# Reward Delivery
def deliver_reward(msg):
    if msg == 1: correctport = rightport
    if msg == 2: correctport = leftport
    ard.write("solenoid " + str(correctport) + " run\r"); 
    index_choice = 'None'
    return(index_choice)

# System Shutdown -- turn off all hardware at the end of a session
def close_all_valves():
    ard.write("solenoid " + str(rightport) + " off\r")
    ard.write("solenoid " + str(leftport) + " off\r")
    tnsy.write("valve " + str(left_valve) + " 1 off\r")
    tnsy.write("valve " + str(right_valve) + " 1 off\r")
    tnsy.write("vialOff " + str(right_valve) + " " + str(odor_vial) + "\r")
    tnsy.write("vialOff " + str(left_valve) + " " + str(odor_vial) + "\r")
    tnsy.write("vialOff " + str(right_valve) + " " + str(blank_vial) + "\r")
    tnsy.write("vialOff " + str(left_valve) + " " + str(blank_vial) + "\r")
    tnsy.write("MFC 1 1 0\r")
    tnsy.write("MFC 1 2 0\r")
    tnsy.write("MFC 2 1 0\r")
    tnsy.write("MFC 2 2 0\r")
