# There are 2 types of darkfield being recorded in the data:
# 1. Darkfield when the laser source is off but the sensor is still open to environment of the recording of the diffraction.
# 2. Darkfield recorded when the a cap is placed on the sensor, so it only records the dark current of the sensor itself, this name file is darkfield_cap.

#%%
import json
import rawpy
import numpy as np  
import tifffile
from pathlib import Path
import matplotlib.pyplot as plt

#################### input json file location ####################
# json file contains the information on the the locations and parameter of the raw data, containing in the same folder
# uni pc
# json_folder_location = r"C:\Users\HDao\Dropbox\2026\Single Slit Diffraction\Single_Slit_Multi_Distance_serious_19_02_26"
# home pc
json_folder_location = r"C:\Users\mnhda\Dropbox\2026\Single Slit Diffraction\Single_Slit_Multi_Distance_serious_19_02_26"


#darkfield image from batch 1
darkfield1_json_name = "darkfield1_20260219_153944_metadata.json"

#darkfield cap image from batch 1
darkfield_cap1_json_name = "darkfield_cap1_20260219_154233_metadata.json"

#darkfield image from batch 6
darkfield6_json_name = "darkfield6_20260219_173306_metadata.json"

#darkfield cap image from batch 6
darkfield_cap6_json_name = "darkfield_cap6_20260219_173449_metadata.json"



#################### Defining functions ####################

# Define a function to normalize an array to between 0 and 1
def normalize_array(arr):
    """Normalize an array to between 0 and 1."""
    arrmin = np.min(arr)
    arrmax = np.max(arr)
    if arrmax - arrmin == 0:
        return np.zeros_like(arr)  # avoid division by zero, return an array of zeros
    else:
        return (arr -arrmin) / (arrmax - arrmin)

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
             'shutter_speed': capture.get('camera_settings', {}).get('shutter_speed', ''),  # shutter speed as string   
             'shutter_speed_float': shutter_speed_to_float(capture.get('camera_settings', {}).get('shutter_speed', '')),  # convert shutter speed to float in seconds
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

# define a function analyze the darkfield image, calculate the mean, median, standard deviation and plot the histogram of the diarkfield image
def analyze_darkfield_image(darkfield_jason_name, file_path):
    """Analyze the darkfield image, calculate the mean, median, standard deviation and plot the histogram of the diarkfield image"""
    # extract the image information from json file
    darkfield_image_info = extract_image_Information(darkfield_jason_name, file_path)
    # loop to extract and analyze all individual images from json file, extract only the green channel
    # start the count for how many images in the json file
    count = 0
    for image_info in darkfield_image_info:
        darkfield_image = extract_image_from_cr2(image_info['cr2_path'])
        mean_intensity = np.mean(darkfield_image)
        median_intensity = np.median(darkfield_image)   
        std_intensity = np.std(darkfield_image)
        print(f"Image Name: {image_info['image_name']}")    
        print(f"Mean Intensity: {mean_intensity}")
        print(f"Median Intensity: {median_intensity}")
        print(f"Standard Deviation: {std_intensity}")
        # print the shutter speed of the image
        print(f"Shutter Speed: {image_info['shutter_speed']} seconds")
        
        # add newly extracted darkfield image to total sum of darkfield images, if it's the first image, initialize the total sum with the first image, otherwise add the new image to the total sum
        if count == 0:
            total_sum = darkfield_image.astype(np.float64)  # initialize total sum with the first image, convert to float64 to avoid overflow
        else:
            total_sum += darkfield_image.astype(np.float64)  # add the new image to the total sum, convert to float64 to avoid overflow
        count += 1
        # find the average of the darkfield images by dividing the total sum by the count of images
    average_darkfield_image = total_sum / count
    # analyze the average darkfield image
    mean_intensity = np.mean(average_darkfield_image)
    median_intensity = np.median(average_darkfield_image)
    std_intensity = np.std(average_darkfield_image)
    print(f"Average Darkfield Image: {darkfield_jason_name}")
    print(f"Mean Intensity: {mean_intensity}")
    print(f"Median Intensity: {median_intensity}")  
    print(f"Standard Deviation: {std_intensity}")   
    return average_darkfield_image


    
############# Main code ##############

#%%
# analyze the darkfield image from batch 1
average_darkfield1 = analyze_darkfield_image(darkfield1_json_name, json_folder_location) 

#%%
# analyze the darkfield cap image from batch 1
average_darkfield_cap1 = analyze_darkfield_image(darkfield_cap1_json_name, json_folder_location) 

#%%
# analyze the darkfield image from batch 6
average_darkfield6 = analyze_darkfield_image(darkfield6_json_name, json_folder_location) 

#%%
# analyze the darkfield cap image from batch 6
average_darkfield_cap6 = analyze_darkfield_image(darkfield_cap6_json_name, json_folder_location) 


### note: from observation of the analysis of the darkfield images of batch 1 and batch 6:

# 1. the mean intensity of batch 1 cap and uncap darkfield images are very similar, to each other. This indicate the there
# minimal contribution of environmetal light to the diffractio recording, and the dark current of the sensor is dominating

# 2. the mean intensity of batch 6 uncapped darkfield image is slightly higher than cap for shutter speed of 1/4 seconds,(2.5 vs 2.3)
# this indicate that there is some contribution of environmental light to the diffractio recording at this shutter speed, 
# this suggest that we should use the uncapped darkfield to correct for images. 

# 3. the standard deviation of the individual darkfield for all batch cap and uncap images is higher than the 
# standard deviation of the average darkfield images from the same batch and capping condition, this is expected.
# we will use the average darkfield for correction to reduce the noise added to the diffraction images.

# conclusion: we will use the average darkfield images from uncapped condition to correct for diffraction
# as it may include the contribution of the enviromental and background light to the diffraction recording.

# some skeptical thoughts on the darkfield analysis:
# 1. The mean of the darkfiel is very low for an uncooled sensor, this my suggest there my have been some hidden subtraction 
# of the darkfield in the camera processing, even though we have set the rawpy postprocess to no_auto_bright=True, no_auto_scale=True, output_bps=16, use_camera_wb=False, use_auto_wb=False,
# user_wb=[1, 1, 1, 1], output_color=rawpy.ColorSpace.raw, gamma=(1, 1), demosaic_algorithm=rawpy.DemosaicAlgorithm.LINEAR, median_filter_passes = 0,
# half_size=False, four_color_rgb=False, there may be some hidden processing in the camera that we are not aware of, which may have subtracted the darkfield from the raw data, this may explain the low mean intensity of the darkfield images.    