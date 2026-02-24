# There are 2 types of darkfield being recorded in the data:
# 1. Darkfield when the laser source is off but the sensor is still open to environment of the recording of the diffraction.
# 2. Darkfield recorded when the a cap is placed on the sensor, so it only records the dark current of the sensor itself, this name file is darkfield_cap.

import json
import rawpy
import numpy as np  
import tifffile
from pathlib import Path
import matplotlib.pyplot as plt

#################### input json file location ####################
# json file contains the information on the the locations and parameter of the raw data, containing in the same folder
json_folder_location = r"C:\Users\HDao\Dropbox\2026\Single Slit Diffraction\Single_Slit_Multi_Distance_serious_19_02_26"

#darkfield image from batch 1
darkfield1_json_name = "darkfield1_20260219_153944_metadata.json"

#darkfield cap image from batch 1
darkfield_cap1_json_name = "darkfield_cap1_20260219_154233_metadata.json"

#darkfield image from batch 6
darkfield6_json_name = "darkfield6_20260219_173306_metadata.json"

#darkfield cap image from batch 6
darkfield_cap6_json_name = "darkfield_cap6_20260219_173449_metadata.json"



#################### Defining functions ####################

# define a function to convert shutter speed string to float in seconds
def shutter_speed_to_float(shutter_speed_str):
    """Convert shutter speed string to float in seconds."""
    if isinstance(shutter_speed_str, str):
        if '/' in shutter_speed_str:
            numerator, denominator = shutter_speed_str.split('/')
            return float(numerator) / float(denominator)
        else:
            return float(shutter_speed_str)
    else:
        return float(shutter_speed_str)

# Define a function to extract image information from json files
def extract_image_Information(json_name, file_path):
    """Extract image names, file paths cr2, file path tif and shutter speeds as list of tuples"""
    file_path = Path(file_path) 
    full_path = file_path/ json_name
    with open(full_path, 'r') as json_file:
        metadata = json.load(json_file)
        captures = metadata.get('captures', [])
        # create a list containing all the information of the image from json file
        # image name
        Image_info = [
            {'image_name': capture.get('image_name', ''),
             'cr2_path': file_path / capture.get('image_name', ''),  # file path of cr2
             'shutter_speed': shutter_speed_to_float(capture.get('camera_settings', {}).get('shutter_speed', '')),  # convert shutter speed to float in seconds
             'iso': capture.get('camera_settings', {}).get('iso', ''),
             'motor_group3': capture.get('motor_groups', {}).get('Group3', {}).get('position', ''), # motor position nested in motor_groups, Group3
             'motor_group4': capture.get('motor_groups', {}).get('Group4', {}).get('position', ''), # motor position nested in motor_groups, Group4
             'motor_group5': capture.get('motor_groups', {}).get('Group5', {}).get('position', ''), # motor position nested in motor_groups, Group5
             'motor_group6': capture.get('motor_groups', {}).get('Group6', {}).get('position', ''), # motor position nested in motor_groups, Group6
             'motor_group7': capture.get('motor_groups', {}).get('Group7', {}).get('position', ''), # motor position nested in motor_groups, Group7
             } for capture in captures

        ]
        return Image_info
    

# define a function to extract a single image from cr2 path raw, extract only the green channel
def extract_image_from_cr2(cr2_path):
    rgb_image = rawpy.imread(str(cr2_path)).postprocess(no_auto_bright=True, no_auto_scale=True, output_bps=16, use_camera_wb=False, use_auto_wb=False,
                          user_wb=[1, 1, 1, 1], output_color=rawpy.ColorSpace.raw, gamma=(1, 1),
                          demosaic_algorithm=rawpy.DemosaicAlgorithm.LINEAR, median_filter_passes = 0,
                          half_size=False, four_color_rgb=False)
    # Crop to nominal dimensions: 5202x3468 -> 5184x3456 (remove 18 height, 8 width; 9 and 4 from each edge)
    rgb_image = rgb_image[9:-9, 4:-4, :]
    green_channel = rgb_image[:, :, 1]  # extract the green channel (index 1)
    return green_channel

# define a function that combine the previous two 

############# Main code ##############



