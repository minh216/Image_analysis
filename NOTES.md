# Project Notes — Single Slit Diffraction Analysis GUI

## Current file
`General_Analysis_ver2.py`

## What the code does
1. Reads `.cr2` RAW images, extracts the green channel
2. Averages darkfield frames, subtracts from images, normalizes by shutter speed
3. Correlates each horizontal profile with a Gaussian kernel to find diffraction troughs
4. Estimates slit width using single-slit formula: `a = (m * λ * L) / x`
5. Plots trough distances and slit width estimates vs vertical pixel position

## GUI status
- Currently just a bare empty `tkinter` window (lines ~47–54)
- `filedialog` is already imported
- No widgets added yet

## Parameters the GUI needs to collect

### Group A — File/folder inputs
| Variable | Description |
|---|---|
| `Folder_location` | Folder with raw images + metadata JSON |
| `Raw_json_file_name` | JSON file for raw images |
| `json_dark_file_name` | JSON file for darkfield images |
| `analysis_folder_location` | Folder to save results |
| `experiement_name` | Subfolder name for this experiment |

### Group B — Physics parameters
| Variable | Default | Description |
|---|---|---|
| `pixel_size` | `4.31e-6` | Camera pixel size in meters |
| `wavelength` | `532e-9` | Laser wavelength in meters |
| `distance_slit_to_screen` | `0.375` | Slit-to-screen distance in meters |

### Group C — Trough-finding parameters
| Variable | Default | Description |
|---|---|---|
| `kernel_size` | `500` | Gaussian kernel size for correlation |
| `sigma` | `25` | Gaussian kernel sigma |
| `X_coordinate` | — | x pixel for profile (or use brightest point) |
| `Y_coordinate` | — | y pixel for profile (or use brightest point) |
| `Upper_vertical_limit` | — | Top of vertical range of interest |
| `Lower_vertical_limit` | — | Bottom of vertical range of interest |

## Next steps for the GUI (in order)

### Step 1 — Add Label + Entry for one field (learn the pattern)
```python
tk.Label(window, text="Folder Location:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
folder_entry = tk.Entry(window, width=50)
folder_entry.grid(row=0, column=1, padx=5, pady=5)
```

### Step 2 — Add a Browse button for folder/file inputs
```python
def browse_folder():
    path = filedialog.askdirectory()
    folder_entry.delete(0, tk.END)
    folder_entry.insert(0, path)

tk.Button(window, text="Browse", command=browse_folder).grid(row=0, column=2, padx=5)
```

### Step 3 — Add a Run button that reads all entries and runs analysis
```python
def run_analysis():
    folder_location = folder_entry.get()
    # ... get the rest of the fields
    window.destroy()

tk.Button(window, text="Run Analysis", command=run_analysis).grid(row=10, column=1, pady=10)
```

### Step 4 — Add all remaining fields (rows 1–9) following the same pattern

### Step 5 — Pre-fill default values for physics/trough params using `entry.insert(0, "default_value")`

### Step 6 — Resize window to fit all widgets (`window.geometry("700x600")` or use `window.resizable(True, True)`)

## To restore context on another machine
Open `General_Analysis_ver2.py` and tell Copilot:
> "Read through this file and NOTES.md — I'm learning to write a tkinter GUI for the analysis parameters."
