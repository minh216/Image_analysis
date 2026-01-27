# Required packages: pip install numpy opencv-python tifffile matplotlib

import json
import pathlib
import numpy as np
import cv2
import tifffile
import matplotlib.pyplot as plt


file_path = pathlib.Path(r"C:\Users\HDao\Dropbox\2026\Single Slit Diffraction\Single_slit_diffraction_serious_26_01_26") # folder containting the JSON file
json_name = "center_1_15_20260126_215858_metadata.json" # name of the JSON file
full_path = file_path / json_name # full path to the JSON file the / operator joins paths can only be used with Path from pathlib

# # Function to extract all image name from the JSON file
# def extract_image_names(json_name, file_path):
#     full_path = file_path / json_name
#     with open(full_path, 'r') as json_file:
#         metadata = json.load(json_file)
#         captures = metadata.get('captures', [])
#         num_images = len(captures)
#         print(f"Number of images in 'captures': {num_images}")
#         image_names = [capture.get('image_name') for capture in captures]
#         tif_image_names = [name.replace(".CR2",".tif") for name in image_names]  # convert .CR2 to .tif
#         return image_names, num_images, tif_image_names
    
    
# # # run the function to extract image names
# # image_names, num_images, tif_image_names = extract_image_names(json_name, file_path)
    
# # print(f"Extracted image names:{tif_image_names}")

# # Function to extract shutter speeds from the JSON file
# def extract_shutter_speeds(json_name, file_path):
#     full_path = file_path / json_name
#     with open(full_path, 'r') as json_file:
#         metadata = json.load(json_file)
#         captures = metadata.get('captures', [])
#         shutter_speeds = [capture.get('camera_settings',{}).get('shutter_speed') for capture in captures]
#         return shutter_speeds
    
# # # run the function to extract shutter speeds
# # shutter_speeds = extract_shutter_speeds(json_name, file_path)
# # print(f"Extracted shutter speeds:{shutter_speeds}") 



def extract_image_exposure_pairs(json_name, file_path):
    """Extract image names and shutter speeds as list of tuples"""
    full_path = file_path / json_name
    with open(full_path, 'r') as json_file:
        metadata = json.load(json_file)
        captures = metadata.get('captures', [])
        
        Image_info = [
            (capture.get('image_name', '').replace('.CR2', '.tif'),
             capture.get('camera_settings', {}).get('shutter_speed'))
            for capture in captures
        ]
        
        return Image_info

# run the function to extract image names and shutter speeds as list of tuples
Image_info = extract_image_exposure_pairs(json_name, file_path)  
print(f"Extracted image names and shutter speeds:{Image_info}")