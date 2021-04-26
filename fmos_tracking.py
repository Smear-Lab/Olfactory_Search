'''
FMOS MODULE
Online Tracking Functions

Written By: Teresa Findley (tmfindley15@gmail.com)
Last Updated: 04.26.2021, Teresa Findley
'''

#     [SET UP]     #

##IMPORTS
##libraries
import cv2
##local modules
#from AUTO_preferences import * #AUTO is used for running automated 2 second trials without a mouse 
from fmos_preferences_bonsai import *

#     [FUNCTIONS]     #

# Partition Quadrants
def calc_partitions():
	start_x = x_min; start_y = y_min; sectionNum = 0
	for x in range (0,x_sections):
        	for y in range(0,y_sections):
			
			end_y = int(start_y +((y_max-y_min)/y_sections))
			end_x = int(start_x + ((x_max-x_min)/x_sections))

			section[sectionNum] = (start_x,start_y,end_x,end_y)
			section_center[sectionNum] = (int((start_x+end_x)/2),int((start_y+end_y)/2))
			start_y = start_y+((y_max-y_min)/y_sections)
			sectionNum = sectionNum + 1
        	start_y = 0
         	start_x = start_x+((x_max-x_min)/x_sections)
	return (section,section_center)

# Detect Mouse's Location within Partitions 
def detect_mouse_partitions(com,section_center,section_occupancy):
	for sectionNum in range(0,len(section)):
		if com[0] >= section[sectionNum][0] and com[0] < section[sectionNum][2]:
			if com[1] >= section[sectionNum][1] and com[1] < section[sectionNum][3]:
				section_occupancy = sectionNum
	return section_occupancy

