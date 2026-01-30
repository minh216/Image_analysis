import json
import pathlib
import numpy as np
import cv2
import tifffile
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import curve_fit

# uni laptop path
# file_path = pathlib.Path(r"C:\Users\HDao\Dropbox\2026\Single Slit Diffraction\Single_slit_diffraction_serious_26_01_26") # folder containting the JSON file

# home pc path
file_path = pathlib.Path(r"C:\Users\mnhda\Dropbox\2026\Single Slit Diffraction\Single_slit_diffraction_serious_26_01_26") # folder containting the JSON file

json_name = "center_1_15_20260126_215858_metadata.json" # name of the JSON file
full_path = file_path / json_name # full path to the JSON file the / operator joins paths can only be used with Path from pathlib

################### Functions ###################

# Define a function to extract image information from JSON file
def extract_image_Information(json_name, file_path):
    """Extract image names, file paths cr2, file path tif and shutter speeds as list of tuples"""
    full_path = file_path / json_name
    with open(full_path, 'r') as json_file:
        metadata = json.load(json_file)
        captures = metadata.get('captures', [])
        
        Image_info = [
            (capture.get('image_name', '').replace('.CR2', '.tif'),
             file_path / capture.get('image_name', ''),
             file_path / capture.get('image_name', '').replace('.CR2', '.tif'),
             capture.get('camera_settings', {}).get('shutter_speed'))
            for capture in captures
        ]
        
        return Image_info
    
# Define a function that extract just the green channel from the tif image_path 
def extract_green_channel(image_path):
    image = tifffile.imread(image_path)  # read the tif image
    green_channel = image[:, :, 1]  # extract the green channel
    return green_channel.astype(np.float64)  # convert to float32 for further analysis  

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


# Define a function to combine all the images in the image_info list of tuples into a single image by 
# combingining their green channels with appropriate normalization based on shutter speed and save the result as a tif file
def combine_images(file_path, json_name, Result_file_name="combined_image.tif"):
    image_info = extract_image_Information(json_name, file_path)
    combined_image = None
    #counter to keep track of number of images combined compared to total number of images
    total_count_images = len(image_info)    
    count = 0
    for tif_name, cr2_path, tif_path, shutter_speed in image_info:
        green_channel = extract_green_channel(tif_path)
        # Normalize the green channel by shutter speed
        shutter_speed_float = shutter_speed_to_float(shutter_speed)
        normalized_channel = green_channel / shutter_speed_float
        if combined_image is None:
            combined_image = normalized_channel
            #counter to keep track of number of images combined compared to total number of images
            count = 1
            print(f"Combining image {count} of {total_count_images}: {tif_name} with shutter speed {shutter_speed} ({shutter_speed_float} seconds)")
        else:
            combined_image += normalized_channel
            count += 1
            print(f"Combining image {count} of {total_count_images}: {tif_name} with shutter speed {shutter_speed} ({shutter_speed_float} seconds)")    
    # save the combined image as a tif file to a new folder called "combined_results" in the file_path
    combined_results_path = file_path / "combined_results"
    combined_results_path.mkdir(exist_ok=True)  # create the folder if it doesn't exist
    tifffile.imwrite(combined_results_path / Result_file_name, combined_image.astype(np.float64))   
    return combined_image.astype(np.float64)


################# Analysis Code ###################

# Combine all center peak of the diffraction images
combined_image = combine_images(file_path, json_name, Result_file_name="center_combined_image.tif") 

# # Display the combined image in natural log scale
# combined_image_log = np.log(combined_image + 1000000)  # add 1 to avoid log(0)
# plt.imshow(combined_image_log, cmap='gray')
# plt.title('Combined Image (Log Scale)')
# plt.colorbar()
# plt.show()

# # Plot the central horizontal cross-section of the combined image in log scale 
# central_row = combined_image[combined_image.shape[0] // 2, :]
# plt.plot(np.log(central_row + 1000000))  # add 1 to avoid log(0)
# plt.title('Central Horizontal Cross-Section of Combined Image (Log Scale)')
# plt.xlabel('Pixel Position')
# plt.ylabel('Intensity (log scale)')
# plt.show()


