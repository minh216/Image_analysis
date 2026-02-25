# This step extract the diffraction images, correct them with darkfield, normalized by the shutter speed, and save the averaged image of the
# stack of repeated images. do this for all 7 batches of recorded single slit diffraction data at 7 different distance of propagation.


import json
import rawpy
import numpy as np  
import tifffile
from pathlib import Path
import matplotlib.pyplot as plt

#################### input json file location ####################
# json file contains the information on the the locations and parameter of the raw data, containing in the same folder
# uni pc
json_folder_location = r"C:\Users\HDao\Dropbox\2026\Single Slit Diffraction\Single_Slit_Multi_Distance_serious_19_02_26"
# home pc
# json_folder_location = r"C:\Users\mnhda\Dropbox\2026\Single Slit Diffraction\Single_Slit_Multi_Distance_serious_19_02_26"

# batch 1
#darkfield image from batch 1
darkfield1_json_name = "darkfield1_20260219_153944_metadata.json"
#darkfield cap image from batch 1
darkfield_cap1_json_name = "darkfield_cap1_20260219_154233_metadata.json"
#center image from batch 1
center1_json_name = "center1_20260219_153041_metadata.json" 

# batch 2
#darkfield image from batch 2
darkfield2_json_name = "darkfield2_20260219_161523_metadata.json"
#darkfield cap image from batch 2   
darkfield_cap2_json_name = "darkfield_cap2_20260219_161759_metadata.json"
#center image from batch 2
center2_json_name = "center2_20260219_160200_metadata.json" 

# batch 3
#darkfield image from batch 3   
darkfield3_json_name = "darkfield3_20260219_163809_metadata.json"
#darkfield cap image from batch 3
darkfield_cap3_json_name = "darkfield_cap3_20260219_163943_metadata.json"
#center image from batch 3
center3_json_name = "center3_20260219_162535_metadata.json"

#batch 4
#darkfield image from batch 4
darkfield4_json_name = "darkfield4_20260219_165522_metadata.json"
#darkfield cap image from batch 4
darkfield_cap4_json_name = "darkfield_cap4_20260219_165908_metadata.json"
#center image from batch 4
center4_json_name = "center4_20260219_164617_metadata.json" 

#batch 5
#darkfield image from batch 5
darkfield5_json_name = "darkfield5_20260219_171630_metadata.json"
#darkfield cap image from batch 5
darkfield_cap5_json_name = "darkfield_cap5_20260219_171758_metadata.json"
#center image from batch 5
center5_json_name = "center5_20260219_170706_metadata.json" 

#batch 6
#darkfield image from batch 6
darkfield6_json_name = "darkfield6_20260219_173306_metadata.json"
#darkfield cap image from batch 6
darkfield_cap6_json_name = "darkfield_cap6_20260219_173449_metadata.json"
#center image from batch 6
center6_json_name = "center6_20260219_172401_metadata.json" 

#batch 7
#darkfield center image from batch 7   
darkfield_center7_json_name = "darkfield_center7_20260219_183052_metadata.json"
#darkfield center cap image from batch 7
darkfield_center_cap7_json_name = "darkfield_center_cap7_20260219_183256_metadata.json"
#darkfield side image from batch 7
darkfield_side7_json_name = "darkfield_side7_20260219_182121_metadata.json"
#darkfield side cap image from batch 7
darkfield_side_cap7_json_name = "darkfield_side_cap7_20260219_182557_metadata.json"
#center image from batch 7
center7_json_name = "center7_20260219_174135_metadata.json"
#left image from batch 7
left7_json_name = "left7_20260219_175818_metadata.json"
#right image from batch 7   
right7_json_name = "right7_20260219_180956_metadata.json"


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
def extract_DFcorrect_average_timenormalized_image(image_json_name, darkfield_json_name, file_path):
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

# extract images from batch 1, apply darkfield correction and time normalization, and get the averaged image for batch 1
average_time_normalized_DFcorrected_image_batch1 = extract_DFcorrect_average_timenormalized_image(center1_json_name, darkfield1_json_name, json_folder_location)
# show the average image for batch 1
plt.imshow(average_time_normalized_DFcorrected_image_batch1, cmap='gray')
plt.show() 
# plot the vertical and horizontal profiles of the average image for batch 1 at the brightest point of the diffraction pattern
plot_V_H_profiles_at_brightest_point(average_time_normalized_DFcorrected_image_batch1)


# extract images from batch 2, apply darkfield correction and time normalization, and get the averaged image for batch 2
average_time_normalized_DFcorrected_image_batch2 = extract_DFcorrect_average_timenormalized_image(center2_json_name, darkfield2_json_name, json_folder_location)
# show the average image for batch 2
plt.imshow(average_time_normalized_DFcorrected_image_batch2, cmap='gray')
plt.show()
# plot the vertical and horizontal profiles of the average image for batch 2 at the brightest point of the diffraction pattern
plot_V_H_profiles_at_brightest_point(average_time_normalized_DFcorrected_image_batch2)


# extract images from batch 3, apply darkfield correction and time normalization, and get the averaged image for batch 3
average_time_normalized_DFcorrected_image_batch3 = extract_DFcorrect_average_timenormalized_image(center3_json_name, darkfield3_json_name, json_folder_location)
# show the average image for batch 3
plt.imshow(average_time_normalized_DFcorrected_image_batch3, cmap='gray')
plt.show()
# plot the vertical and horizontal profiles of the average image for batch 3 at the brightest point of the diffraction pattern
plot_V_H_profiles_at_brightest_point(average_time_normalized_DFcorrected_image_batch3)  


# extract images from batch 4, apply darkfield correction and time normalization, and get the averaged image for batch 4
average_time_normalized_DFcorrected_image_batch4 = extract_DFcorrect_average_timenormalized_image(center4_json_name, darkfield4_json_name, json_folder_location)
plt.imshow(average_time_normalized_DFcorrected_image_batch4, cmap='gray')
plt.show()
# plot the vertical and horizontal profiles of the average image for batch 4 at the brightest point of the diffraction pattern
plot_V_H_profiles_at_brightest_point(average_time_normalized_DFcorrected_image_batch4)


# extract images from batch 5, apply darkfield correction and time normalization, and get the averaged image for batch 5
average_time_normalized_DFcorrected_image_batch5 = extract_DFcorrect_average_timenormalized_image(center5_json_name, darkfield5_json_name, json_folder_location)
plt.imshow(average_time_normalized_DFcorrected_image_batch5, cmap='gray')
plt.show()
# plot the vertical and horizontal profiles of the average image for batch 5 at the brightest point of the diffraction pattern
plot_V_H_profiles_at_brightest_point(average_time_normalized_DFcorrected_image_batch5)


# extract images from batch 6, apply darkfield correction and time normalization, and get the averaged image for batch 6
average_time_normalized_DFcorrected_image_batch6 = extract_DFcorrect_average_timenormalized_image(center6_json_name, darkfield6_json_name, json_folder_location)
plt.imshow(average_time_normalized_DFcorrected_image_batch6, cmap='gray')
plt.show()  
# plot the vertical and horizontal profiles of the average image for batch 6 at the brightest point of the diffraction pattern
plot_V_H_profiles_at_brightest_point(average_time_normalized_DFcorrected_image_batch6)


# extract center images from batch 7, apply center darkfield correction and time normalization, and get the averaged  center image for batch 7
average_time_normalized_DFcorrected_image_center7 = extract_DFcorrect_average_timenormalized_image(center7_json_name, darkfield_center7_json_name, json_folder_location)
plt.imshow(average_time_normalized_DFcorrected_image_center7, cmap='gray')
plt.show()
# plot the vertical and horizontal profiles of the average image for batch 7 at the brightest point of the diffraction pattern
plot_V_H_profiles_at_brightest_point(average_time_normalized_DFcorrected_image_center7) 


# extract left images from batch 7, apply side darkfield correction and time normalization, and get the averaged left image for batch 7
average_time_normalized_DFcorrected_image_left7 = extract_DFcorrect_average_timenormalized_image(left7_json_name, darkfield_side7_json_name, json_folder_location)
plt.imshow(average_time_normalized_DFcorrected_image_left7, cmap='gray')
plt.show()
# plot the vertical and horizontal profiles of the average image for batch 7 at the brightest point of the diffraction pattern
plot_V_H_profiles_at_brightest_point(average_time_normalized_DFcorrected_image_left7)   


# extract right images from batch 7, apply side darkfield correction and time normalization, and get the averaged right image for batch 7
average_time_normalized_DFcorrected_image_right7 = extract_DFcorrect_average_timenormalized_image(right7_json_name, darkfield_side7_json_name, json_folder_location)
plt.imshow(average_time_normalized_DFcorrected_image_right7, cmap='gray')
plt.show()  
# plot the vertical and horizontal profiles of the average image for batch 7 at the brightest point of the diffraction pattern
plot_V_H_profiles_at_brightest_point(average_time_normalized_DFcorrected_image_right7)  


# save these averaged images as tif files in a folder named "analyzed_images" in the same location as the json file
analyzed_images_folder = Path(json_folder_location) / "analyzed_images"
analyzed_images_folder.mkdir(exist_ok=True) # create the folder if it does not exist    
tifffile.imwrite(analyzed_images_folder / "average_time_normalized_DFcorrected_image_batch1.tif", average_time_normalized_DFcorrected_image_batch1.astype(np.float32))
tifffile.imwrite(analyzed_images_folder / "average_time_normalized_DFcorrected_image_batch2.tif", average_time_normalized_DFcorrected_image_batch2.astype(np.float32))
tifffile.imwrite(analyzed_images_folder / "average_time_normalized_DFcorrected_image_batch3.tif", average_time_normalized_DFcorrected_image_batch3.astype(np.float32))
tifffile.imwrite(analyzed_images_folder / "average_time_normalized_DFcorrected_image_batch4.tif", average_time_normalized_DFcorrected_image_batch4.astype(np.float32))
tifffile.imwrite(analyzed_images_folder / "average_time_normalized_DFcorrected_image_batch5.tif", average_time_normalized_DFcorrected_image_batch5.astype(np.float32))
tifffile.imwrite(analyzed_images_folder / "average_time_normalized_DFcorrected_image_batch6.tif", average_time_normalized_DFcorrected_image_batch6.astype(np.float32))
tifffile.imwrite(analyzed_images_folder / "average_time_normalized_DFcorrected_image_center7.tif", average_time_normalized_DFcorrected_image_center7.astype(np.float32))
tifffile.imwrite(analyzed_images_folder / "average_time_normalized_DFcorrected_image_left7.tif", average_time_normalized_DFcorrected_image_left7.astype(np.float32))
tifffile.imwrite(analyzed_images_folder / "average_time_normalized_DFcorrected_image_right7.tif", average_time_normalized_DFcorrected_image_right7.astype(np.float32))  
