import json
import pathlib

file_path = pathlib.Path(r"C:\Users\HDao\Dropbox\2026\Single Slit Diffraction\Single_slit_diffraction_serious_26_01_26") # folder containting the JSON file
json_name = "center_1_15_20260126_215858_metadata.json" # name of the JSON file

full_path = file_path / json_name # full path to the JSON file the / operator joins paths can only be used with Path from pathlib


print(f"Loading JSON file from: {full_path}") # f string for formatted output

with open(full_path, 'r') as json_file: # open the JSON file in read mode the with is to automatically close the file after reading
    metadata = json.load(json_file) # load the JSON data into a Python dictionary
    print("JSON file loaded successfully.") # confirmation message
    captures = metadata.get('captures', []) # get the 'captures' list from the metadata dictionary
    num_images = len(captures) # count the number of images in the captures list
    print(f"Number of images in 'captures': {num_images}") # print the number
    capture_key = list(captures[0].keys()) if captures else None # get all key from the first capture if captures is not empty
    print(f"All key in the first capture: {capture_key}") # print the keys




