#%%
import json
import pathlib
import numpy as np
import cv2
import rawpy
import tifffile
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import curve_fit

# uni laptop path
file_path = pathlib.Path(r"C:\Users\HDao\Dropbox\2026\Single Slit Diffraction\Single_slit_diffraction_serious_26_01_26") # folder containting the JSON file
image_test_file_path_CR2 = pathlib.Path(r"C:\Users\HDao\Dropbox\2026\Single Slit Diffraction\test pixel of raw conversion 2\Extra_Pixel_test.CR2")
image_test_file_path_jpg = pathlib.Path(r"C:\Users\HDao\Dropbox\2026\Single Slit Diffraction\test pixel of raw conversion 2\Extra_Pixel_test.jpg")

# home pc path
# file_path = pathlib.Path(r"C:\Users\mnhda\Dropbox\2026\Single Slit Diffraction\Single_slit_diffraction_serious_26_01_26") # folder containting the JSON file
# image_test_file_path_CR2 = pathlib.Path(r"C:\Users\mnhda\Dropbox\2026\Single Slit Diffraction\test pixel of raw conversion 2\Extra_Pixel_test.CR2")
# image_test_file_path_jpg = pathlib.Path(r"C:\Users\mnhda\Dropbox\2026\Single Slit Diffraction\test pixel of raw conversion 2\Extra_Pixel_test.jpg")

# JSON file names for center, left, right images and dark field images for 1/15 seconds and 1.3 seconds setup
center_json_name = "center_1_15_20260126_215858_metadata.json" # name of the JSON file
left_json_name = "left_1__3_20260126_225055_metadata.json"
right_json_name = "right_1__3_20260126_231335_metadata.json"
darkfield_1_15_json_name = "dark_field_1_15_20260126_233157_metadata.json"
darkfield_1__3_json_name = "dark_field_1__3_20260126_233456_metadata.json"

# camera parameters
pixel_size_mm = 4.31  # pixel size in microns
Camera_pixel_X = 3456  # camera pixel in X direction
Camera_pixel_Y = 5184  # camera pixel in Y direction

################### Functions ###################

# Define a function to normalize an array to between 0 and 1
def normalize_array(arr):
    """Normalize an array to between 0 and 1."""
    arrmin = np.min(arr)
    arrmax = np.max(arr)
    if arrmax - arrmin == 0:
        return np.zeros_like(arr)  # avoid division by zero, return an array of zeros
    else:
        return (arr -arrmin) / (arrmax - arrmin)

# Define a function to extract image information from JSON file
def extract_image_Information(json_name, file_path):
    """Extract image names, file paths cr2, file path tif and shutter speeds as list of tuples"""
    full_path = file_path / json_name
    with open(full_path, 'r') as json_file:
        metadata = json.load(json_file)
        captures = metadata.get('captures', [])
        # create a list of tuples containing image name, file path of cr2, file path of tif and shutter speed for each capture
        Image_info = [
            (
                capture.get('image_name', ''),  # image name
                file_path / capture.get('image_name', ''),  # file path of cr2
                file_path / (capture.get('image_name', '').replace('.CR2', '.tif')),  # file path of tif
                capture.get('camera_settings', {}).get('shutter_speed', '')  # shutter speed nested in camera_settings
            )
            for capture in captures 
        ]
        
        return Image_info
    
# Define a function to convert CR2 raw to tif using rawpy full bit depth without without any alteration to the raw data, and save the tif file in the same folder as the CR2 file with the same name but with .tif extension
def convert_cr2_to_tif(cr2_path, tif_folder_name="tif_images"):
    tif_folder_path = cr2_path.parent / tif_folder_name
    tif_folder_path.mkdir(parents=True, exist_ok=True)
    tif_path = tif_folder_path / (cr2_path.stem + ".tif")
    with rawpy.imread(str(cr2_path)) as raw:
        rgb_tif = raw.postprocess(no_auto_bright=True, no_auto_scale=True, output_bps=16, use_camera_wb=False, use_auto_wb=False, 
                             user_wb=[1, 1, 1, 1], output_color=rawpy.ColorSpace.raw, gamma=(1, 1),
                             demosaic_algorithm=rawpy.DemosaicAlgorithm.LINEAR, median_filter_passes = 0,
                             half_size=False, four_color_rgb=False)  # keep the full bit depth without any alteration to the raw data
        # Crop to nominal dimensions: 5202x3468 -> 5184x3456 (remove 18 height, 12 width; 9 and 6 from each edge)
        rgb_tif = rgb_tif[9:-9, 6:-6, :]
        tifffile.imwrite(tif_path, rgb_tif)  # save the image as a tif file with full bit depth
        return tif_path, rgb_tif
    
# Define a function to convert CR2 raw to tif using raw.raw_image_visible (just check it seem even the raw_image_visible still return image with extra pixels, I think I will just extract raw.raw_image and try to have my own demosaicing separatedly)
def convert_cr2_to_tif_raw_image(cr2_path, tif_folder_name="tif_images"):
    # making the folder path
    tif_folder_path = cr2_path.parent / tif_folder_name
    # create the folder if it doesn't exists
    tif_folder_path.mkdir(parents=True, exist_ok=True)
    # making the tif path from CR2 path
    tif_path = tif_folder_path / (cr2_path.stem + ".tif")
    with rawpy.imread(str(cr2_path)) as raw:
        raw_image_visible = raw.raw_image.astype(np.float32)  # get the raw image as float32
        tifffile.imwrite(tif_path, raw_image_visible)  # save the image as a tif file with full bit depth
        return tif_path, raw_image_visible
    

        
# Define a function to extract just the green channel from floate image generated by convert_cr2_to_tif function output
def extract_green_channel_from_float_image(float_image):
    green_channel = float_image[:, :, 1]  # extract the green channel
    return green_channel.astype(np.float32)  # convert to float64 for further analysis  

# Define a function that extract just the green channel from the tif image_path 
def extract_green_channel(image_path):
    image = tifffile.imread(image_path)  # read the tif image
    green_channel = image[:, :, 1]  # extract the green channel
    return green_channel.astype(np.float32)  # convert to float32 for further analysis  

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

# Define a function to extract motor postion from json files for center, left and right images. motor position is in "motor_group", each motor 
# is "Group3", "Group4", etc. Group3 is Camera X axis, Group4 is Camera Y axis
def extract_motor_position(json_name, file_path):
    full_path = file_path / json_name
    with open(full_path, 'r') as json_file:
        metadata = json.load(json_file)
        Cam_X_position = metadata.get('motor_group', {}).get('Group3', 0)  # X axis motor position
        Cam_Y_position = metadata.get('motor_group', {}).get('Group4', 0)  # Y axis motor position
        return Cam_X_position, Cam_Y_position


################# Analysis Code ###################

# #%%
# # test the function by printing the image information for the dark field 1/15 seconds setup
# darkfield_1_15_image_info = extract_image_Information(darkfield_1_15_json_name, file_path)
# # print the information from darkfield tupple   
# for tif_name, cr2_path, tif_path, shutter_speed in darkfield_1_15_image_info:
#     print(f"Image Name: {tif_name}")
#     print(f"CR2 Path: {cr2_path}")
#     print(f"TIF Path: {tif_path}")
#     print(f"Shutter Speed: {shutter_speed}")
#     print("--------------------------------------------------")

############### Step 0.5: determine the extra pixel in the raw and crop to actual image size of 5184 x 3456 ###############
#%%
# load the test raw CR2 file and the preview JPG file to determine the location of the extra pixels
test_cr2_path = image_test_file_path_CR2
test_jpg_path = image_test_file_path_jpg
test_pixel_tif_path, test_pixel_tif_array = convert_cr2_to_tif(test_cr2_path, tif_folder_name="test_pixel_tif_images") # convert the test CR2 file to tif and save in a new folder called "test_pixel_tif_images" in the same directory as the test CR2 file
#plot the horizontal profile of test_pixel_tif_array for each of the RGB channels to make sure the RGB match with tif image
plt.plot(test_pixel_tif_array[test_pixel_tif_array.shape[0] // 2, :, 0], label='TIF Red Channel', marker='^', color='red', alpha=0.2, markersize=1)
plt.plot(test_pixel_tif_array[test_pixel_tif_array.shape[0] // 2, :, 1], label='TIF Green Channel', marker='^', color='green', alpha=0.2, markersize=1)
plt.plot(test_pixel_tif_array[test_pixel_tif_array.shape[0] // 2, :, 2], label='TIF Blue Channel', marker='^', color='blue', alpha=0.2, markersize=1)
plt.title('Horizontal Profile of RGB Channels for Test Pixel TIF')  
plt.legend()
plt.show()  

#%%
test_pixel_jpg = cv2.imread(str(test_jpg_path))  # read the test JPG file
test_pixel_jpg = cv2.cvtColor(test_pixel_jpg, cv2.COLOR_BGR2RGB)  # convert from BGR to RGB color space
#plot the horizontal profile of test_pixel_jpg for each of the RGB channels to make sure the RGB match with tif image
plt.plot(test_pixel_jpg[test_pixel_jpg.shape[0] // 2, :, 0], label='JPG Red Channel', marker='o', color='red', alpha=0.2, markersize=1)   
plt.plot(test_pixel_jpg[test_pixel_jpg.shape[0] // 2, :, 1], label='JPG Green Channel', marker='o', color='green', alpha=0.2, markersize=1)
plt.plot(test_pixel_jpg[test_pixel_jpg.shape[0] // 2, :, 2], label='JPG Blue Channel', marker='o', color='blue', alpha=0.2, markersize=1)
plt.title('Horizontal Profile of RGB Channels for Test Pixel JPG')
plt.legend()
plt.show()

#%%
# comparing the size of the extracted tif and jpg
print(f"Test Pixel TIF shape: {test_pixel_tif_array.shape}")  # should be (5184, 3456, 3) if no extra pixels
print(f"Test Pixel JPG shape: {test_pixel_jpg.shape}")  # should be (5184, 3456, 3) if no extra pixels


# extract the green channel from the test pixel tif and test pixel jpg
test_pixel_tif_green = extract_green_channel_from_float_image(test_pixel_tif_array)
test_pixel_jpg_green = extract_green_channel_from_float_image(test_pixel_jpg)   
# overplot the horizontal profile of the green channel for both the test pixel tif and test pixel jpg to determine the location of the extra pixels on the left or right side of the image, triangle for tif and circle for jpg, 50% transparency
plt.plot(normalize_array(test_pixel_tif_green[test_pixel_tif_green.shape[0] // 2, :]), label='TIF Green Channel', marker='^', alpha=0.5)
plt.plot(normalize_array(test_pixel_jpg_green[test_pixel_jpg_green.shape[0] // 2, :]), label='JPG Green Channel', marker='o', alpha=0.5)
plt.title('Horizontal Profile of Green Channel for Test Pixel TIF and JPG')
plt.xlabel('Pixel Position')
plt.ylabel('Intensity')
plt.legend()
plt.show()


#%%
# convert CR2 file to tif using the raw_image method and compare with the postprocess method
# convert the test CR2 file and display the bayer pattern
test_pixel_tif_raw_visible_path, test_pixel_tif_raw_visible_array = convert_cr2_to_tif_raw_image(test_cr2_path, tif_folder_name="test_pixel_tif_raw_visible_images")
# check size the raw visible tif
print(f"test_pixel_tif_raw_visible_array shape: {test_pixel_tif_raw_visible_array.shape}")  # should be (5184, 3456) if no extra pixels
# show the bayer pattern
plt.imshow(test_pixel_tif_raw_visible_array, cmap='gray')
plt.title('Bayer Pattern of Test Pixel TIF Raw Visible')
plt.xlabel('Pixel Position')
plt.ylabel('Pixel Position')
plt.colorbar()
plt.show()  


############### Step 1: Extracting image from CR2 to tif################

#%%
############## Step 1.1: Extracting dark field images for 1/15 seconds setup and analyzing their statistics ###############

# Extracting image from CR2 to tif for all dark field images for 1/15 seconds setup while analyzing the dark field image statistics
darkfield_1_15_image_info = extract_image_Information(darkfield_1_15_json_name, file_path)
print("Extracting CR2 to tif and analyzing dark field 1/15 seconds individual image statistics:")
# Initialize combined_darkfield
combined_darkfield = None
# Check the statistics for each dark field image and averaging them to a final average dark field image for 1/15 seconds setup
for tif_name, cr2_path, tif_path, shutter_speed in darkfield_1_15_image_info:
    # convert and save tif file from cr2 file
    tif_path, rgb_tif = convert_cr2_to_tif(cr2_path,"darkfield_1_15_tif_images")
    green_channel = extract_green_channel_from_float_image(rgb_tif)
    # Accumulate the images
    if combined_darkfield is None:
        combined_darkfield = green_channel
    else:
        combined_darkfield += green_channel 

    print(f"Image: {tif_name}, Shutter Speed: {shutter_speed}")
    print(f"Mean: {np.mean(green_channel)}")   
    print(f"Median: {np.median(green_channel)}")
    print(f"Standard Deviation: {np.std(green_channel)}")
    print("--------------------------------------------------")

# Average the dark field images for 1/15 seconds setup and save to the same folder with the name "average_non_linear_darkfield_1_15.tif"
average_non_linear_darkfield_1_15 = np.float32(combined_darkfield / len(darkfield_1_15_image_info))

# Analyze the statistic for the average dark field image for 1/15 seconds setup
print("Average Non-linear Dark Field 1/15 seconds statistics:")
print(f"Mean: {np.mean(average_non_linear_darkfield_1_15)}")      
print(f"Median: {np.median(average_non_linear_darkfield_1_15)}")
print(f"Standard Deviation: {np.std(average_non_linear_darkfield_1_15)}")   
tifffile.imwrite(file_path / "darkfield_1_15_tif_images" / "average_non_linear_darkfield_1_15.tif", average_non_linear_darkfield_1_15.astype(np.float64))

# Normalize the average dark field to between 0 and 1 and show the log scale image
plt.imshow(np.log10(normalize_array(average_non_linear_darkfield_1_15) + 1), cmap='gray')  # add 1 to avoid log(0)
plt.title('Average Non-linear Dark Field 1/15 seconds (Log Scale)')
plt.colorbar()


#%%
############## Step 1.2: Extracting dark field images for 1.3 seconds setup and analyzing their statistics ###############

# Extracting image from CR2 to tif for all dark field images for 1.3 seconds setup while analyzing the dark field image statistics
darkfield_1__3_image_info = extract_image_Information(darkfield_1__3_json_name, file_path)
print("Extracting CR2 to tif and analyzing dark field 1.3 seconds individual image statistics:")
# Initialize combined_darkfield
combined_darkfield = None
# Check the statistics for each dark field image and averaging them to a final average dark field image for 1.3 seconds setup
for tif_name, cr2_path, tif_path, shutter_speed in darkfield_1__3_image_info:
    # convert and save tif file from cr2 file
    tif_path, rgb_tif = convert_cr2_to_tif(cr2_path,"darkfield_1__3_tif_images")
    green_channel = extract_green_channel_from_float_image(rgb_tif)
    # Accumulate the images
    if combined_darkfield is None:
        combined_darkfield = green_channel
    else:
        combined_darkfield += green_channel 

    print(f"Image: {tif_name}, Shutter Speed: {shutter_speed}")
    print(f"Mean: {np.mean(green_channel)}")   
    print(f"Median: {np.median(green_channel)}")
    print(f"Standard Deviation: {np.std(green_channel)}")
    print("--------------------------------------------------")

# Average the dark field images for 1.3 seconds setup and save to the same folder with the name "average_non_linear_darkfield_1__3.tif"
average_non_linear_darkfield_1__3 = np.float32(combined_darkfield / len(darkfield_1__3_image_info))

# Analyze the statistic for the average dark field image for 1.3 seconds setup
print("Average Non-linear Dark Field 1.3 seconds statistics:")
print(f"Mean: {np.mean(average_non_linear_darkfield_1__3)}  ")    
print(f"Median: {np.median(average_non_linear_darkfield_1__3)}")
print(f"Standard Deviation: {np.std(average_non_linear_darkfield_1__3)}")   
tifffile.imwrite(file_path / "darkfield_1__3_tif_images" / "average_non_linear_darkfield_1__3.tif", average_non_linear_darkfield_1__3.astype(np.float64))

# Normalize the average dark field to between 0 and 1 and show the log scale image
plt.imshow(np.log10(normalize_array(average_non_linear_darkfield_1__3) + 1), cmap='gray')  # add 1 to avoid log(0)
plt.title('Average Non-linear Dark Field 1.3 seconds (Log Scale)')
plt.colorbar()  

# ############### Analysis of dark field images ###############

# #%%

# # Analyze individual dark field images for 1/15 seconds setup
# darkfield_1_15_image_info = extract_image_Information(darkfield_1_15_json_name, file_path)
# # print tif path
# print("Dark Field 1/15 seconds individual image paths:")
# for tif_name, cr2_path, tif_path, shutter_speed in darkfield_1_15_image_info:
#     print(f"Image: {tif_name}, TIF Path: {tif_path}, Shutter Speed: {shutter_speed}")   
# print("Dark Field 1/15 seconds individual image statistics:")  
# for tif_name, cr2_path, tif_path, shutter_speed in darkfield_1_15_image_info:
#     green_channel = extract_green_channel(tif_path)
#     print(f"Image: {tif_name}, Shutter Speed: {shutter_speed}")
#     print(f"Mean: {np.mean(green_channel)}")      
#     print(f"Median: {np.median(green_channel)}")
#     print(f"Standard Deviation: {np.std(green_channel)}")
#     print("--------------------------------------------------")



# # Combine dark field images for 1/15 seconds setup
# combined_darkfield_1_15 = average_non_linear_darkfield(file_path, darkfield_1_15_json_name, Result_file_name="average_non_linear_darkfield_1_15.tif")  

# # Check dark field image 1/15 statistics: means, median, std    
# print("Dark Field 1/15 seconds statistics:")
# print(f"Mean: {np.mean(combined_darkfield_1_15)}")      
# print(f"Median: {np.median(combined_darkfield_1_15)}")
# print(f"Standard Deviation: {np.std(combined_darkfield_1_15)}")




####################### Step 2: Extract combine and average diffraction images ###############


#%%
#
# # Combine all center peak of the diffraction images
# combined_image = combine_images(file_path, center_json_name, Result_file_name="center_averaged_image.tif") 

# #%%
# # Display the combined image in natural log scale
# combined_image_log = np.log(combined_image + 100000)  # add 1 to avoid log(0)
# plt.imshow(combined_image_log, cmap='gray')
# plt.title('Averaged Image (Log Scale)')
# plt.colorbar()
# plt.show()

# #%%
# # Plot the central horizontal cross-section of the averaged image in log scale 
# central_row = combined_image[combined_image.shape[0] // 2, :]
# plt.plot(np.log(central_row + 100000))  # add 1 to avoid log(0)
# plt.title('Central Horizontal Cross-Section of Averaged Image (Log Scale)')
# plt.xlabel('Pixel Position')
# plt.ylabel('Intensity (log scale)')
# plt.show() 

# #%%
# # Combine all left peak of the diffraction images
# left_json_name = "left_1__3_20260126_225055_metadata.json"
# combined_left_image = combine_images(file_path, left_json_name, Result_file_name="left_averaged_image.tif")

# #%%
# # Combine all right peak of the diffraction images
# right_json_name = "right_1__3_20260126_231335_metadata.json"
# combined_right_image = combine_images(file_path, right_json_name, Result_file_name="right_averaged_image.tif


    
    