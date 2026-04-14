# Main analysis use code from the steps in the other functions in this folder to do general analysis of the data. This is the main entry point for the analysis.

#%%
# imports needed for the analysis
import json
import rawpy
import numpy as np  
import tifffile
from pathlib import Path
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

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

################## input the parameters for the single slit analysis ####################
# input parameters
pixel_size = 4.31e-6  # pixel size in meters (4.31 micrometers)
wavelength = 532e-9  # wavelength of the green laser in meters (532 nm)
distance_slit_to_screen = 0.375  # distance from the slit to the screen in meters (0.375 meters)

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

# define a function to plot the vertical and horizontal profiles at the brightest point of the diffraction pattern, with an x marking the position of the brightest point in the profiles, grid on and the value of the brightest point marked on the profiles

def plot_V_H_profiles_at_brightest_point(image):
    brightest_point = np.unravel_index(np.argmax(image), image.shape)
    brightest_value = np.max(image) 
    x_brightest = brightest_point[1]  # x coordinate of the brightest point 
    y_brightest = brightest_point[0]  # y coordinate of the brightest point
    vertical_profile, horizontal_profile = V_H_profiles(image, x_brightest, y_brightest)
    # plot the vertical and horizontal profiles 
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)    
    plt.plot(vertical_profile)
    # add an x marking the position of the brightest point in the vertical profile and display the value of the brightest point on the plot
    plt.plot(y_brightest, vertical_profile[y_brightest], 'rx') 
    # add text to display the value of the brightest point on the plot, with an offset from the x marking
    plt.text(y_brightest + 10, vertical_profile[y_brightest], f'Brightest Point Value: {brightest_value:.2f}', color='red')      
    plt.title('Vertical Profile at Brightest Point')    
    plt.xlabel('Pixel Position (y)')
    plt.ylabel('Intensity (a.u.)')      
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(horizontal_profile)    
    # add an x marking the position of the brightest point in the horizontal profile and display the value of the brightest point on the plot
    plt.plot(x_brightest, horizontal_profile[x_brightest], 'rx')  
    # add text to display the value of the brightest point on the plot, with an offset from the x marking
    plt.text(x_brightest + 10, horizontal_profile[x_brightest], f'Brightest Point Value: {brightest_value:.2f}', color='red')         
    plt.title('Horizontal Profile at Brightest Point')    
    plt.xlabel('Pixel Position (x)')        
    plt.ylabel('Intensity (a.u.)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()  

# define a function to make a 1D gaussian kernel
def gaussian_kernel_1d(size, sigma):
    if size % 2 == 0:
        size += 1  # make size odd to preserve the center of the kernel 
    """Generate a 1D Gaussian kernel."""
    ax = np.linspace(-(size / 2), size / 2, size)
    kernel = np.exp(-0.5 * (ax / sigma) ** 2)
    return kernel / np.sum(kernel)  

# define a function to find the distance between the central peaks and the troughs indexes in correlation_troughs_indices and combined with the integer m for the order into a tupple together with distance by sorted by closest 
def find_trough_distances_and_orders(horizontal_profile_combined, pixel_size, wavelength, distance_slit_to_screen, kernel_size=500, sigma=25):
    # find the peaks and troughs from the horizontal profile of the combined image using by correlating with a guassian kernel
    # kernel_size = 2001  # size of the Gaussian kernel(larger is better but slower too small and risk finding peaks in noise) need to be odd for the kernel to be symmetric
    # sigma = 27  # standard deviation of the Gaussian kernel (too small then increase the risk of finding peaks in noise, too large the accruracy of finding the peaks and troughs will decrease)
    kernel = gaussian_kernel_1d(kernel_size, sigma)  # create the Gaussian kernel
    inverted_profile = -horizontal_profile_combined  # invert the profile to find troughs as peaks
    correlation_peak = np.correlate(horizontal_profile_combined, kernel, mode='same')  # correlate the inverted profile with the Gaussian kernel
    correlation_trough = np.correlate(inverted_profile, kernel, mode='same')  # correlate the inverted profile with the Gaussian kernel
    peak_indices, _ = find_peaks(correlation_peak, distance=400) # use find peaks to get the peaks location from the correlation result of the horizontal profile with the Gaussian kernel, distance is set to 400 to avoid finding peaks that are too close to each other which are likely to be noise
    trough_indices, _ = find_peaks(correlation_trough, distance=400)  # use find peaks to get the troughs location from the correlation result witht he inverted profile
    # initialize empty lists to store the trough distances and orders for left and right troughs
    trough_distances_orders_left = []
    trough_distances_orders_right = []
    # identify the central peak index as the one with the maximum value in the list of peak_indices:
    central_peak_index = peak_indices[np.argmax(horizontal_profile_combined[peak_indices])] 
    # for loop to go through each trough index and sort them into left and right troughs based on their position relative to the central peak, and calculate the distance from the central peak to each trough and determine the order of the trough on the left and right
    # the order of the trough is determined by the distance from the central peak to the trough, with the closest trough being the first order (m=1), the second closest trough being the second order (m=2), and so on. The distance from the central peak to each trough is calculated in meters using the pixel size, and then used to estimate the slit width using the single slit diffraction formula: a = (m * wavelength * distance_slit_to_screen) / x, where m is the order of the minima, wavelength is the wavelength of the laser, distance_slit_to_screen is the distance from the slit to the screen, and x is the distance from the central peak to the minima (trough) in meters.   
    for trough_index in trough_indices:
        distance_in_pixels = abs(trough_index - central_peak_index)
        distance_in_meters = distance_in_pixels * pixel_size
        # find angle of diffraction theta as the arctan of the distance from the central peak to the trough divided by the distance from the slit to the screen
        theta = np.arctan(distance_in_meters / distance_slit_to_screen) 
        if trough_index < central_peak_index:
            # add the new found distance to the list of trough distances
            trough_distances_orders_left.append((trough_index, distance_in_meters, theta))
        else:
            trough_distances_orders_right.append((trough_index, distance_in_meters, theta))
    
    # sort the trough distances and orders by distance from the central peak    
    trough_distances_orders_left.sort(key=lambda x: abs(x[1]))  # sort by distance
    trough_distances_orders_right.sort(key=lambda x: abs(x[1]))  # sort by distance  
    # assign the order based on the sorted distance for left troughs
    for i, trough_info in enumerate(trough_distances_orders_left):
        trough_index, distance_in_meters, theta = trough_info
        order = -(i + 1)  # order is negative for left troughs
        slit_width_estimate = (abs(order) * wavelength * distance_slit_to_screen) / distance_in_meters
        trough_distances_orders_left[i] = (trough_index, distance_in_meters, order, slit_width_estimate, theta)    
    # assign the order based on the sorted distance for right troughs
    for i, trough_info in enumerate(trough_distances_orders_right):
        trough_index, distance_in_meters, theta = trough_info
        order = i + 1  # order is positive for right troughs
        slit_width_estimate = (order * wavelength * distance_slit_to_screen) / distance_in_meters
        trough_distances_orders_right[i] = (trough_index, distance_in_meters, order, slit_width_estimate, theta)
    # Analyze the trough distances for the first order left and right troughs and output the mean value and the uncertainty 
    average_estimated_slit_width_from_first_order = []
    # get the average of the slit width estimates from the first order troughs on the left and right
    average_estimated_slit_width_from_first_order = (trough_distances_orders_left[0][3] + trough_distances_orders_right[0][3])/2 # slit width estimate from the first order trough on the left
    # calculate the uncertainty as the absolute difference between the slit width estimates from the first order troughs on the left and right
    uncertainty_estimated_slit_width_from_first_order = abs(trough_distances_orders_left[0][3] - trough_distances_orders_right[0][3]) / 2 # uncertainty is half the difference between the slit width estimates from the first order troughs
    print(f"Estimated slit width from the first order trough on the left: {trough_distances_orders_left[0][3]:.6e} meters, or {trough_distances_orders_left[0][3]*1e6:.2f} micrometers")
    print(f"Estimated slit width from the first order trough on the right: {trough_distances_orders_right[0][3]:.6e} meters, or {trough_distances_orders_right[0][3]*1e6:.2f} micrometers")
    print(f"Average estimated slit width from the first order troughs: {average_estimated_slit_width_from_first_order:.6e} meters, or {average_estimated_slit_width_from_first_order*1e6:.2f} micrometers")
    print(f"Uncertainty in estimated slit width from the first order troughs: {uncertainty_estimated_slit_width_from_first_order:.6e} meters, or {uncertainty_estimated_slit_width_from_first_order*1e6:.2f} micrometers")      
    return central_peak_index,trough_distances_orders_left, trough_distances_orders_right, average_estimated_slit_width_from_first_order, uncertainty_estimated_slit_width_from_first_order    

# define a function that take in the combined image, the vertical range of interest, use the find_trough_distances_and_orders function to find all the slit estimate and uncertainty for all horizontal in the area of interest and output the average slit width estimate and uncertainty across all the horizontal profiles in the area of interest
# the function will also output the slit width estimates and uncertainties for each horizontal profile in the area of interest in a list of tuples together with the vertical position of the horizontal profile in the combined image
def analyze_horizontal_profiles_in_area_of_interest(combined_image, vertical_range, pixel_size, wavelength, distance_slit_to_screen, kernel_size=500, sigma=25):
    slit_width_estimates_uncertainties = []  # list to store slit width estimates and uncertainties for each horizontal profile
    trough_distances_left_list = [] # list to store trough distances and vertical index for left troughs for each horizontal profile
    trough_distances_right_list = [] # list to store trough distances and vertical index for right troughs for each horizontal profile
    central_peak_indices = [] # list to store central peak indices for each horizontal profile
    horizontal_profile_list = [] # list to store horizontal profiles for each vertical index in the area of interest
    for y in range(vertical_range[0], vertical_range[1]):
       
       horizontal_profile = combined_image[y, :]
       central_peak_index, trough_distances_orders_left, trough_distances_orders_right, average_estimated_slit_width_from_first_order, uncertainty_estimated_slit_width_from_first_order = find_trough_distances_and_orders(horizontal_profile, pixel_size, wavelength, distance_slit_to_screen, kernel_size, sigma)
       slit_width_estimates_uncertainties.append((y, average_estimated_slit_width_from_first_order, uncertainty_estimated_slit_width_from_first_order))
       trough_distances_left_list.append((y, trough_distances_orders_left))
       trough_distances_right_list.append((y, trough_distances_orders_right))
       central_peak_indices.append((y, central_peak_index))
       horizontal_profile_list.append((y, horizontal_profile))
       print(f"Vertical incdex of the horizontal profile: {y})")
       print() # just a line bread for better readability in the output

    # calculate the average slit width estimate and uncertainty across all horizontal profiles in the area of interest
    average_slit_width_estimate_across_profiles = np.mean([estimate for _, estimate, _ in slit_width_estimates_uncertainties])
    average_uncertainty_across_profiles = np.mean([uncertainty for _, _, uncertainty in slit_width_estimates_uncertainties])

    # standard deviation of all left and right slit width estimates from all horizontal profiles in the area of interest
    all_slit_width_estimates = [estimate for _, estimate, _ in slit_width_estimates_uncertainties]
    std_slit_width_estimates_across_profiles = np.std(all_slit_width_estimates)
    
    # standard error from standard deviation and number of profiles
    standard_error_slit_width_estimates_across_profiles = std_slit_width_estimates_across_profiles / np.sqrt(len(slit_width_estimates_uncertainties))

    print("################ final results for the area of interest ################")
    print(f"Average slit width estimate across all horizontal profiles in the area of interest: {average_slit_width_estimate_across_profiles:.6e} meters, or {average_slit_width_estimate_across_profiles*1e6:.2f} micrometers")
    print(f"Average uncertainty across all horizontal profiles in the area of interest: {average_uncertainty_across_profiles:.6e} meters, or {average_uncertainty_across_profiles*1e6:.2f} micrometers")      
    print(f"Standard deviation of slit width estimates across all horizontal profiles in the area of interest: {std_slit_width_estimates_across_profiles:.6e} meters, or {std_slit_width_estimates_across_profiles*1e6:.2f} micrometers")
    print(f"Standard error of slit width estimates across all horizontal profiles in the area of interest: {standard_error_slit_width_estimates_across_profiles:.6e} meters, or {standard_error_slit_width_estimates_across_profiles*1e6:.2f} micrometers") 
    
    return horizontal_profile_list, central_peak_indices, slit_width_estimates_uncertainties, average_slit_width_estimate_across_profiles, average_uncertainty_across_profiles, trough_distances_left_list, trough_distances_right_list  


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
# plot the vertical and horizontal profiles at the brightest point of the extracted image with grid on and value of the brightest point marked on the profiles
plot_V_H_profiles_at_brightest_point(average_time_normalized_DFcorrected_image) 

# %%
#### Step3: analyze the horizontal profile with single slit analyis to find the slit width

# define the vertical range of interest for the analysis based on the vertical profile at the brightest point
vertical_range_of_interest = (brightest_point[0]-200, brightest_point[0]+200)  
print(f"Vertical range of interest for single slit analysis: {vertical_range_of_interest}") 
# print break line for better readability in the output
print() 


# analyze the horizontal profiles in the area of interest to find the slit width estimates and uncertainties for each profile, and the average slit width estimate and uncertainty across all profiles in the area of interest
horizontal_profile_list, central_peak_indices, slit_width_estimates_uncertainties, average_slit_width_estimate_across_profiles, average_uncertainty_across_profiles, trough_distances_left_list, trough_distances_right_list = analyze_horizontal_profiles_in_area_of_interest(average_time_normalized_DFcorrected_image, vertical_range_of_interest, pixel_size, wavelength, distance_slit_to_screen, kernel_size=500, sigma=25)  

#%%
# ploting the results
# plot the y postion as x and distance from central peaks of the first order troughs on the left as y with error bars from the uncertainty in the slit width estimates from the first order troughs using trough_distances_orders_left_list
y_positions_left = [item[0] for item in trough_distances_left_list]  # extract the vertical index y from trough_distances_left_list
y_posittion_right = [item[0] for item in trough_distances_right_list]  # extract the vertical index y from trough_distances_right_list  
first_order_trough_distances_left = [item[1][0][1] for item in trough_distances_left_list]  # extract the distance from the central peak to the first order trough on the left from trough_distances_left_list
first_order_trough_distance_right = [item[1][0][1] for item in trough_distances_right_list]  # extract the distance from the central peak to the first order trough on the right from trough_distances_right_list
slit_width_estimates_left = [item[1][0][3] for item in trough_distances_left_list]  # extract the slit width estimates from the first order trough on the left from trough_distances_left_list
slit_width_estimates_right = [item[1][0][3] for item in trough_distances_right_list]  # extract the slit width estimates from the first order trough on the right from trough_distances_right_list  
# extract from slit_width_estimates_uncertainties the average_estimated_slit_width_from_first_order for each vertical index
y_position_left_right = [item[0] for item in slit_width_estimates_uncertainties]  # extract the vertical index y from slit_width_estimates_uncertainties    
slit_width_estimate_left_right = [item[1] for item in slit_width_estimates_uncertainties]  # extract the average_estimated_slit_width_from_first_order for each vertical index from slit_width_estimates_uncertainties
slit_width_estimate_left_right_uncertainty = [item[2] for item in slit_width_estimates_uncertainties]  # extract the average_uncertainty_across_profiles for each vertical index from slit_width_estimates_uncertainties    
# plot the trough distances vs y position for left and right troughs no error bars for now (the ylable on the left) and the slitwidth_estimates from right and left first order troughs(ylable on the right)
fig,ax1 = plt.subplots(figsize=(20, 12))
color = 'tab:blue'
ax1.set_xlabel('Vertical Position (y index)')   
ax1.set_ylabel('Distance from Central Peak to First Order Trough (mm)', color=color)
ax1.plot(y_positions_left, np.array(first_order_trough_distances_left)*1e3, color='blue', marker='o', label='Left Troughs')
ax1.plot(y_posittion_right, np.array(first_order_trough_distance_right)*1e3, color='orange', marker='o', label='Right Troughs')
ax1.tick_params(axis='y', labelcolor=color) 
ax1.legend(loc='upper left')
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:green' 
#limit the y axis of ax2 to be between 0 and 200 microns
ax2.set_ylim(0, 200) 
ax2.set_ylabel('Slit Width Estimate from First Order Troughs (microns)', color=color)  # we already handled the x-label with ax1
ax2.plot(y_positions_left, np.array(slit_width_estimates_left)*1e6, color='blue', marker='x', label='Slit Width Estimate from Left Troughs')
ax2.plot(y_posittion_right, np.array(slit_width_estimates_right)*1e6, color='orange', marker='x', label='Slit Width Estimate from Right Troughs')
ax2.errorbar(y_position_left_right, np.array(slit_width_estimate_left_right)*1e6, yerr=np.array(slit_width_estimate_left_right_uncertainty)*1e6, color='maroon', marker='+', label='Average Slit Width Estimate from Left and Right Troughs')
#draw a horizontal line for the slit width estimate at 80 microns for reference
ax2.axhline(y=80, color='black', linestyle='--', label='Reference Slit Width (80 microns)')
ax2.legend(loc='upper right')       
#draw a vertical line for the vertical position of the brightest point for reference
ax1.axvline(x=brightest_point[0], color='red', linestyle='--', label='Brightest Point Vertical Position')
ax1.legend(loc='upper left')    
# add finer gridlines for better visibility
ax1.grid(which='both', linestyle='--', linewidth=0.5)

plt.title('Distance from Central Peak to First Order Troughs and Slit Width Estimates vs Vertical Position')
plt.tight_layout()  # otherwise the right y-label is slightly clipped   
plt.show()
#%%
# plot the distance horizonal profile and the mark the location of the central peak and the first order troughs on the left and right
left_trough_index = trough_distances_left_list[0][1][0][0]  # extract the index of the first order trough on the left from trough_distances_left_list
right_trough_index = trough_distances_right_list[0][1][0][0]  # extract the index of the first order trough on the right from trough_distances_right_list
central_peak_index = central_peak_indices[0][1]  # extract the index of the central peak from central_peak_indices
horizontal_profile =horizontal_profile_list[0][1]  # extract the horizontal profile from horizontal_profile_list
plt.figure(figsize=(12, 6))
plt.plot(horizontal_profile, label='Horizontal Profile at Brightest Point')
plt.plot(central_peak_index, horizontal_profile[central_peak_index], 'rx', label='Central Peak')  # mark the central peak with a red x
plt.plot(left_trough_index, horizontal_profile[left_trough_index], 'go', label='First Order Trough (Left)')  # mark the first order trough on the left with a green circle
plt.plot(right_trough_index, horizontal_profile[right_trough_index], 'mo', label='First Order Trough (Right)')  # mark the first order trough on the right with a magenta circle
plt.title('Horizontal Profile at Brightest Point with Central Peak and First Order Troughs Marked')
plt.xlabel('Pixel Position (x)')    
plt.ylabel('Intensity (a.u.)')
plt.legend()
plt.grid(True)
plt.show()





