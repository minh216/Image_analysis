# Required packages: pip install numpy opencv-python tifffile matplotlib

import json
import pathlib
import numpy as np
import cv2
import tifffile
import matplotlib.pyplot as plt
import scipy.ndimage
from scipy.optimize import curve_fit

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

# # run the function to extract image names and shutter speeds as list of tuples
image_info = extract_image_Information(json_name, file_path)  
# print(f"Extracted image names and shutter speeds:{image_info}")

# Define a function that extract just the green channel from the tif image_path 

def extract_green_channel(image_path):
    image = tifffile.imread(image_path)  # read the tif image
    green_channel = image[:, :, 1]  # extract the green channel
    return green_channel.astype(np.float64)  # convert to float32 for further analysis  

# # Test the function on the first image info tuple
green_channel_test = extract_green_channel(image_info[0][2])
# # Display the green channel

# print(f"Data type: {green_channel_test.dtype}")
# print(f"type: {type(green_channel_test)}")
# print(f"Shape: {green_channel_test.shape}")
# print(f"Min value: {green_channel_test.min()}")
# print(f"Max value: {green_channel_test.max()}")
# print({image_info[0]})

# plt.imshow(green_channel_test, cmap='gray')
# plt.show()

# define a function to plot the vertical and horizontal profile of an image with type in position x and y in pixels
def V_H_profiles(image, x, y):
    vertical_profile = image[:, x]
    horizontal_profile = image[y, :]
    return vertical_profile, horizontal_profile

# # test the plotting function
# Vertical_P, Horizontal_P = V_H_profiles(green_channel_test, x=green_channel_test.shape[1]//2, y=green_channel_test.shape[0]//2)

# define a function to make a gaussian kernel
def gaussian_kernel(size, sigma):
    """Generate a Gaussian kernel."""
    ax = np.linspace(-(size / 2), size / 2, size)
    print(f"type of ax: {ax.dtype}")
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)

# # test the gaussian kernel function
# kernel = gaussian_kernel(size=20, sigma=5)
# plt.imshow(kernel, cmap='gray')
# plt.title('Gaussian Kernel')
# plt.colorbar()      
# plt.show()  

# define a function that find the brightest spot in the image and return its coordinates
def find_brightest_spot(image):
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(image)
    return max_loc, max_val  # returns (x, y) coordinates of the brightest spot

# # test the find_brightest_spot function
# brightest_spot, max_value = find_brightest_spot(green_channel_test)
# print(f"Brightest spot coordinates: {brightest_spot}, Max value: {max_value}")   

# # define a function to find the brightest spot and the horizontal profile through it. then fit a gaussian function to the profile to find the appropriate sigma. the input include the image and the kernal size for gaussian
# def analyze_brightest_spot(image, kernel_size=1500):
#     brightest_spot, max_value = find_brightest_spot(image)
#     x, y = brightest_spot
#     vertical_profile, horizontal_profile = V_H_profiles(image, x, y)
#     # cut the horizontal profile to half the kernel size on each side of the brightest spot
#     half_size = kernel_size // 2
#     start = x - half_size
#     end = x + half_size + 1
#     horizontal_profile_small = horizontal_profile[start:end]

#     # Fit a Gaussian to the horizontal profile to estimate sigma

#     def gaussian(x, amp, mean, sigma, offset):
#         return amp * np.exp(-((x - mean) ** 2) / (2 * sigma ** 2)) + offset

#     x_data = np.arange(len(horizontal_profile_small))
#     initial_guess = [max_value, x, kernel_size / 2, np.min(horizontal_profile_small)]
    
#     try:
#         popt, _ = curve_fit(gaussian, x_data, horizontal_profile_small, p0=initial_guess)
#         fitted_sigma = popt[2]
#     except RuntimeError:
#         fitted_sigma = None  # fitting failed

#     # over plot the horizontal profile and the fitted gaussian
#     plt.figure()
#     plt.plot(horizontal_profile_small, label='Horizontal Profile', marker='o')
#     if fitted_sigma is not None:
#         plt.plot(x_data, gaussian(x_data, *popt), label='Fitted Gaussian')
#     plt.legend()
#     plt.show()

#     return {
#         'brightest_spot': brightest_spot,
#         'max_value': max_value,
#         'horizontal_profile': horizontal_profile,
#         'fitted_sigma': fitted_sigma
#     }

# # test the analyze_brightest_spot function
# analysis_result = analyze_brightest_spot(green_channel_test, kernel_size=1500)
# print(f"Analysis Result: {analysis_result}")    


# script to test out the procedure to find the appropriate gaussian kernel sigma for close approximation to the central peak

# find the brightest spot in the green channel test image
brightest_spot, max_value = find_brightest_spot(green_channel_test)
x, y = brightest_spot   
# get the horizontal profile through the brightest spot
vertical_profile, horizontal_profile = V_H_profiles(green_channel_test, x, y)
# define the kernel size
kernel_size = 1500
# cut the horizontal profile to half the kernel size on each side of the brightest spot
half_size = kernel_size // 2
start = x - half_size   
end = x + half_size + 1
horizontal_profile_small = horizontal_profile[start:end]
# Fit a Gaussian to the horizontal profile to estimate sigma
# define the gaussian function
def gaussian(x, amp, mean, sigma, offset):
    return amp * np.exp(-((x - mean) ** 2) / (2 * sigma ** 2)) + offset 
x_data = np.arange(len(horizontal_profile_small))
print(x_data)
initial_guess = [ max_value, kernel_size / 2,kernel_size / 4, np.min(horizontal_profile_small)]
initial_gaussian_guess = gaussian(x_data, *initial_guess)
# plot the initial guess and the horizontal profile small
plt.figure()
plt.plot(horizontal_profile_small, label='Horizontal Profile', marker='o')
plt.plot(x_data, initial_gaussian_guess, label='Initial Gaussian Guess')
plt.legend()
plt.show()
# fit the gaussian to the horizontal profile small
try:
    popt, _ = curve_fit(gaussian, x_data, horizontal_profile_small, p0=initial_guess)
    fitted_sigma = popt[2]  
    print(f"Fitted sigma: {fitted_sigma}")
except RuntimeError:
    fitted_sigma = None  # fitting failed   
# over plot the horizontal profile and the fitted gaussian
plt.figure()    

if fitted_sigma is not None:
    plt.plot(x_data, gaussian(x_data, *popt), label='Fitted Gaussian')
    plt.plot( horizontal_profile_small, alpha=0.5, label='Horizontal Profile', marker='o', markersize = 0.5)
plt.legend()
plt.show()  