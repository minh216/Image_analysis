# Main analysis use code from the steps in the other functions in this folder to do general analysis of the data. This is the main entry point for the analysis.

#%%
# imports needed for the analysis
import json
import rawpy
import numpy as np  
import tifffile
from pathlib import Path
import matplotlib.pyplot as plt

#################### input json file location ####################
# json file contains the information on the the locations and parameter of the raw data, containing in the same folder

# folder location that contain the raw file (images and dark fields) and the metadata json file
json_folder_location = r"C:\Users\HDao\Dropbox\2026\Single Slit Diffraction\Single_Slit_Multi_Distance_serious_19_02_26"

# raw images json file name
json_file_name = "center1_20260219_153041_metadata.json"

# raw dark fields json file name
json_dark_file_name = "darkfield1_20260219_153944_metadata.json"

################### input the analysis folder location and folder name for saving the converted images and the analysis results ####################
# folder location for saving the converted images and the analysis results
analysis_folder_location = r"C:\Users\HDao\Dropbox\2026\Single Slit Diffraction\Image_analysis\Analysis_results"
# experiment name 
experiement_name = "Single_Slit_Multi_Distance_serious_19_02_26"

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

# define a function to extract image, and normalize it by the shutter speed, summing all images in the stack
# then average the summed image by the number of images in the stack, and apply darkfield correction by subtracting the darkfield image, and return the corrected averaged image
def extract_DFcorrect_average_timenormalized_image(image_json_name, darkfield_json_name, file_path, analysis_folder_path, dark_field_folder_path, corrected_image_folder_path):
    # extract image information from json file
    image_info = extract_image_Information(image_json_name, file_path)
    darkfield_info = extract_image_Information(darkfield_json_name, file_path)
    # extract first images from the image stack and darkfield stact to initialize the summed image and darkfield image
    first_image = extract_image_from_cr2(image_info[0]['cr2_path']) 
    first_darkfield = extract_image_from_cr2(darkfield_info[0]['cr2_path'])
    # initialize the summed image and darkfield image   
    summed_image = np.zeros_like(first_image, dtype=np.float64)  # initialize summed image with the same shape as the first image, but with float64 data type for precision
    summed_darkfield = np.zeros_like(first_darkfield, dtype=np.float64)  # initialize summed darkfield with the same shape as the first darkfield, but with float64 data type for precision
    # loop to extract all darkfield images in the stack from json file, and normalized them by the shutter speed, and sum them together
    for darkfield in darkfield_info:
        darkfield_image = extract_image_from_cr2(darkfield['cr2_path'])  # extract darkfield image from cr2 path
        summed_darkfield += darkfield_image  # sum the darkfield images together we don't time normalized the darkfield due to analysis in step 0.5 show dark field is independent of the shutter speed for these range of shutter speed.
        # add visibility to the loop by printing the progress
        print(f"Processing darkfield image: {darkfield['image_name']} - Progress: {darkfield_info.index(darkfield) + 1}/{len(darkfield_info)}") 
    # find the average darkfield image by dividing the summed darkfield image by the number of images in the darkfield stack
    average_darkfield = summed_darkfield / len(darkfield_info)
    # save the average darkfield image as a tif file in the dark field folder
    average_darkfield_tif_path = dark_field_folder_path / f"{darkfield_json_name}_average_darkfield.tif"
    tifffile.imwrite(average_darkfield_tif_path, average_darkfield.astype(np.float32))
    # print progress of finishing finding the average darkfield image
    print(f"Finished finding average darkfield image for {darkfield_json_name}")
    # loop to extract all images in the stack from json file, and normalized them by the shutter speed, and sum them together
    for image in image_info:
        image_stack = extract_image_from_cr2(image['cr2_path'])  # extract image from cr2 path
        # darfield correction for each image
        DF_corrected_image_stack = image_stack - average_darkfield  # apply darkfield correction by subtracting the average darkfield image from the image stack
        time_normalized_DF_corrected_image_stack = DF_corrected_image_stack / image['shutter_speed_float']  # normalize image by shutter speed
        summed_image += time_normalized_DF_corrected_image_stack  # sum the normalized images together
        # add visibility to the loop by printing the progress
        print(f"Processing image: {image['image_name']} - Progress: {image_info.index(image) + 1}/{len(image_info)}")     
    # find the average image by dividing the summed image by the number of images in the stack
    average_time_normalized_DFcorrected_image = summed_image / len(image_info)
    # ensure the average image is non-negative by setting any negative values to zero
    average_time_normalized_DFcorrected_image[average_time_normalized_DFcorrected_image < 0] = 0
    # save the average time normalized darkfield corrected image as a tif file in the corrected image folder
    average_time_normalized_DFcorrected_image_tif_path = corrected_image_folder_path / f"{image_json_name}_average_time_normalized_DFcorrected_image.tif"
    tifffile.imwrite(average_time_normalized_DFcorrected_image_tif_path, average_time_normalized_DFcorrected_image.astype(np.float32))
    # print progress of finishing finding the average image
    print(f"Finished finding average time normalized darkfield corrected image for {image_json_name}")  
    return average_time_normalized_DFcorrected_image

# define a function to extract the vertical and horizontal profile of an image with type in position x and y in pixels
def V_H_profiles(image, x, y):
    vertical_profile = image[:, x]
    horizontal_profile = image[y, :]
    return vertical_profile, horizontal_profile 

# define a function to plot the vertical and horizontal profiles at the brightest point of the diffraction pattern, with an x marking the position of the brightest point in the profiles
def plot_V_H_profiles_at_brightest_point(image):
    brightest_point = np.unravel_index(np.argmax(image), image.shape)
    x_brightest = brightest_point[1]  # x coordinate of the brightest point 
    y_brightest = brightest_point[0]  # y coordinate of the brightest point
    vertical_profile, horizontal_profile = V_H_profiles(image, x_brightest, y_brightest)
    # plot the vertical and horizontal profiles 
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)    
    plt.plot(vertical_profile)
    # add an x marking the position of the brightest point in the vertical profile
    plt.plot(y_brightest, vertical_profile[y_brightest], 'rx')  
    plt.title('Vertical Profile at Brightest Point')    
    plt.xlabel('Pixel Position (y)')
    plt.ylabel('Intensity (a.u.)')      
    plt.subplot(1, 2, 2)
    plt.plot(horizontal_profile)    
    # add an x marking the position of the brightest point in the horizontal profile
    plt.plot(x_brightest, horizontal_profile[x_brightest], 'rx')    
    plt.title('Horizontal Profile at Brightest Point')    
    plt.xlabel('Pixel Position (x)')        
    plt.ylabel('Intensity (a.u.)')
    plt.tight_layout()
    plt.show()  


############# Main code ##############

#%%
#### Step 1: create folder for the analysis

# create folder for saving the converted images and the analysis results with the experiment name
analysis_folder = Path(analysis_folder_location) / experiement_name
analysis_folder.mkdir(parents=True, exist_ok=True)  
# create folder for extracted darfield
dark_field_folder = analysis_folder / "Extracted_Darkfield"
dark_field_folder.mkdir(parents=True, exist_ok=True)    
# create folder for saving the converted images corrected with darkfield correction and time normalization
corrected_image_folder = analysis_folder / "Corrected_Images"   
corrected_image_folder.mkdir(parents=True, exist_ok=True)  

#%%
#### Step 2: extract the average darkfield image, and the average time normalized darkfield corrected image, and save them as tif files in the corresponding folders
# extract the average darkfield image, and the average time normalized darkfield corrected image
average_time_normalized_DFcorrected_image = extract_DFcorrect_average_timenormalized_image(json_file_name, json_dark_file_name, json_folder_location, analysis_folder, dark_field_folder, corrected_image_folder)

#%%
# find the brightest point of the extracted image and show the image with a yellow marker at the brightest point
brightest_point = np.unravel_index(np.argmax(average_time_normalized_DFcorrected_image), average_time_normalized_DFcorrected_image.shape)
plt.imshow(average_time_normalized_DFcorrected_image, cmap='gray')  
plt.plot(brightest_point[1], brightest_point[0], '+', color='red')  # plot a yellow marker at the brightest point
plt.title('Average Time Normalized Darkfield Corrected Image with Brightest Point Marked')      
plt.axis('off')
plt.show()  
# plot the vertical and horizontal profiles at the brightest point of the extracted image
plot_V_H_profiles_at_brightest_point(average_time_normalized_DFcorrected_image) 




# %%
