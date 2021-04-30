Code for running Smear Lab olfactory search assay using Arduino, Python, and Bonsai. 

ARDUINO: All hardware is run using custom Arduino programs run on a Teensy board and a Mega2560 board. These programs must be loaded before running behavioral assay program in python (see FMOS Protocol -Setup). 

PYTHON: Python programs should be downloaded and stored in the same folder in your working directory. All scripts are modules containing functions except for the trainers, mastercode, and preferences files. The preferences file is the only one that should be edited unless re-structuring the experiment. 

BONSAI: Video capture and real time tracking uses the open source computer vision program Bonsai (https://github.com/bonsai-rx). There is a nose, head, and body tracking program available (headtracking) and a center-of-mass tracking program (COM). The nose tracking depends on red paint applied to the center of the mouse's head. Bonsai sends coordinates to python in real time via OSC. This is specific to an individual's computer and the OSC library will need to be installed and set up with local addresses for communication to work. 

DEEPLABCUT: We improve our tracking quality by tracking mouse coordinates offline using Deeplabcut (https://github.com/DeepLabCut/DeepLabCut). Custom networks available upon request. 
