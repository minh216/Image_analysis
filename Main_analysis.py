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

center_json_name = "center_1_15_20260126_215858_metadata.json" # name of the JSON file
left_json_name = "left_1__3_20260126_225055_metadata.json"
right_json_name = "right_1__3_20260126_231335_metadata.json"
darkfield_1_15_json_name = "dark_field_1_15_20260126_233157_metadata.json"
darkfield_1__3_json_name = "dark_field_1__3_20260126_233456_metadata.json"

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

# Define a function to combine all the dark field for specific setup. since the darkfield was determined to be none lineared with
# the shutter speed and comprised mostly of some noise or bias that is indepedent of the shutter speed, we will simply average the dark field images without normalizing by shutter speed

def average_non_linear_darkfield(file_path, json_name, Result_file_name="average_non_linear_darkfield_image.tif"):
    image_info = extract_image_Information(json_name, file_path)
    combined_image = None
    #counter to keep track of number of images combined compared to total number of images
    total_count_images = len(image_info)    
    count = 0
    for tif_name, cr2_path, tif_path, shutter_speed in image_info:
        green_channel = extract_green_channel(tif_path)
        # Since dark field is non linear with shutter speed, we do not normalize by shutter speed
        normalized_channel = green_channel
        # average the dark field images
        if combined_image is None:  
            combined_image = normalized_channel
            #counter to keep track of number of images combined compared to total number of images
            count = 1
            print(f"Combining image {count} of {total_count_images}: {tif_name}")
        else:
            combined_image += normalized_channel
            count += 1
            print(f"Combining image {count} of {total_count_images}: {tif_name}")    
        average_non_linear_darkfield = combined_image / total_count_images
    # save the averaged image as a tif file to a new folder called "combined_results" in the file_path
    average_non_linear_darkfield_results_path = file_path / "combined_results"
    average_non_linear_darkfield_results_path.mkdir(exist_ok=True)  # create the folder if it doesn't exist
    tifffile.imwrite(average_non_linear_darkfield_results_path / Result_file_name, average_non_linear_darkfield.astype(np.float64))   
    return average_non_linear_darkfield.astype(np.float64)

# Define a function to combine all the images in the image_info list of tuples into a single image by 
# combingining their green channels (after minusing the darkfield) with appropriate normalization based on shutter speed and save the result as a tif file.
# also normalized by total number of images combined to match with other images with different number of shot and shutter speeds
def combine_images(file_path, json_name, Result_file_name="averaged_image.tif",average_darkfield=None):
    image_info = extract_image_Information(json_name, file_path)
    combined_image = None
    #counter to keep track of number of images combined compared to total number of images
    total_count_images = len(image_info)    
    count = 0
    for tif_name, cr2_path, tif_path, shutter_speed in image_info:
        green_channel = extract_green_channel(tif_path)
        # subtract darkfield before normalizing by shutter speed
        if average_darkfield is not None:
            green_channel -= average_darkfield
            # ensure no negative values after darkfield subtraction
            green_channel = np.clip(green_channel, a_min=0, a_max=None) 
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
    average_image = combined_image / total_count_images
    # save the combined image as a tif file to a new folder called "combined_results" in the file_path
    average_results_path = file_path / "average_results"
    average_results_path.mkdir(exist_ok=True)  # create the folder if it doesn't exist
    tifffile.imwrite(average_results_path / Result_file_name, average_image.astype(np.float64))   
    return average_image.astype(np.float64)




################# Analysis Code ###################
#%%
# Combine dark field images for 1/15 seconds setup
combined_darkfield_1_15 = average_non_linear_darkfield(file_path, darkfield_1_15_json_name, Result_file_name="average_non_linear_darkfield_1_15.tif")  

# Check dark field image 1/15 statistics: means, median, std    
print("Dark Field 1/15 seconds statistics:")
print(f"Mean: {np.mean(combined_darkfield_1_15)}")      
print(f"Median: {np.median(combined_darkfield_1_15)}")
print(f"Standard Deviation: {np.std(combined_darkfield_1_15)}")

#%%
# Combine dark field images for 1.3 seconds setup
combined_darkfield_1__3 = average_non_linear_darkfield(file_path, darkfield_1__3_json_name, Result_file_name="average_non_linear_darkfield_1__3.tif")  

# Check dark field image 1.3 statistics: means, median, std
print("Dark Field 1.3 seconds statistics:")
print(f"Mean: {np.mean(combined_darkfield_1__3)}")      
print(f"Median: {np.median(combined_darkfield_1__3)}")
print(f"Standard Deviation: {np.std(combined_darkfield_1__3)}")

#%%
# Combine all center peak of the diffraction images
combined_image = combine_images(file_path, center_json_name, Result_file_name="center_averaged_image.tif") 

#%%
# Display the combined image in natural log scale
combined_image_log = np.log(combined_image + 100000)  # add 1 to avoid log(0)
plt.imshow(combined_image_log, cmap='gray')
plt.title('Averaged Image (Log Scale)')
plt.colorbar()
plt.show()

#%%
# Plot the central horizontal cross-section of the averaged image in log scale 
central_row = combined_image[combined_image.shape[0] // 2, :]
plt.plot(np.log(central_row + 100000))  # add 1 to avoid log(0)
plt.title('Central Horizontal Cross-Section of Averaged Image (Log Scale)')
plt.xlabel('Pixel Position')
plt.ylabel('Intensity (log scale)')
plt.show() 

#%%
# Combine all left peak of the diffraction images
left_json_name = "left_1__3_20260126_225055_metadata.json"
combined_left_image = combine_images(file_path, left_json_name, Result_file_name="left_averaged_image.tif")

#%%
# Combine all right peak of the diffraction images
right_json_name = "right_1__3_20260126_231335_metadata.json"
combined_right_image = combine_images(file_path, right_json_name, Result_file_name="right_averaged_image.tif")

# # stitch the left, center and right combined images together based on motor positions and pixel size
# pixel_size_mm = 4.31  # pixel size in microns
# # extract motor postion from json files for center, left and right images
# def extract_motor_position(json_name, file_path):
#     full_path = file_path / json_name
#     with open(full_path, 'r') as json_file:
#         metadata = json.load(json_file)
#         motor_position = metadata.get('motor_position', 0)
#         return motor_position