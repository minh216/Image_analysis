# Image_analysis
Single slit diffraction image analysis - image stacking and processing

## Installation

Install required packages:

```bash
pip install numpy opencv-python tifffile matplotlib scipy
```

## Overview

This project processes single slit diffraction pattern images captured with a DSLR camera. It combines multiple exposures with shutter speed normalization and optional darkfield correction.

## Main Functions

### `extract_image_Information(json_name, file_path)`
Extracts image metadata from JSON files created during capture.

**Returns:** List of tuples containing:
- TIF image name
- CR2 file path
- TIF file path  
- Shutter speed

### `extract_green_channel(image_path)`
Extracts the green channel from RGB TIFF images for analysis.

**Returns:** Green channel as float64 numpy array

### `shutter_speed_to_float(shutter_speed_str)`
Converts shutter speed strings (e.g., "1/15" or "1.3") to float values in seconds.

### `average_non_linear_darkfield(file_path, json_name, Result_file_name)`
Averages darkfield images without shutter speed normalization.

**Note:** Darkfield was found to be dominated by read noise (~15 counts) rather than thermal dark current, making it negligible compared to signal levels (45,000 counts at peaks).

**Parameters:**
- `file_path`: Path to image directory
- `json_name`: JSON metadata filename
- `Result_file_name`: Output filename

**Returns:** Averaged darkfield image as float64 array

### `combine_images(file_path, json_name, Result_file_name, average_darkfield=None)`
Combines multiple images with shutter speed normalization and optional darkfield subtraction.

**Process:**
1. Extracts green channel from each image
2. Optionally subtracts darkfield
3. Normalizes by shutter speed (intensity per second)
4. Averages all normalized images
5. Saves result to "average_results" folder

**Parameters:**
- `file_path`: Path to image directory
- `json_name`: JSON metadata filename
- `Result_file_name`: Output filename
- `average_darkfield`: Optional pre-calculated darkfield image

**Returns:** Averaged combined image as float64 array

## Usage

```python
# Process darkfield images
darkfield = average_non_linear_darkfield(file_path, "dark_field_metadata.json", "darkfield.tif")

# Combine signal images with normalization
combined = combine_images(file_path, "center_metadata.json", "center_averaged.tif", average_darkfield=darkfield)
```

## Output

Processed images are saved as 64-bit floating-point TIFF files to preserve the full dynamic range after averaging and normalization.

## Notes

- All print statements include `flush=True` for real-time progress updates
- Darkfield correction is optional since darkfield contribution is typically <0.1% of signal
- Images are normalized by total count for consistent intensity across different exposure settings
