import os
import joblib
from logging_config import log
from art import tprint


def dsave(data, category, name=None, path=".janus_result.pkl"):
    if os.path.exists(path):
        try:
            result = joblib.load(path)
        except EOFError:  # Handle corruption
            log.info("Warning: result.pkl is corrupted. Recreating the file...")
            os.remove(path)  # Delete the corrupted file
            result = {"pra": {}, "pra_percomplex": {}, "pr_auc": {}, "tmp": {}, "input": {}}
    else:
        log.progress(f"'{path}' does not exist. Creating a new result structure.")
        result = {"pra": {}, "pra_percomplex": {}, "pr_auc": {}, "tmp": {}, "input": {}}

    if category not in result:
        result[category] = {}

    if name:
        result[category][name] = data
    else:
        result[category] = data

    joblib.dump(result, path)


def dload(category, name=None, path=".janus_result.pkl"):
    if os.path.exists(path):
        result = joblib.load(path)
        category_data = result.get(category, {})
        if name:
            return category_data.get(name, {})  # ← return empty dict instead of None
        return category_data
    return {}  # ← fallback if file doesn't exist


def deep_update(source, overrides):
    """Recursively update the source dict with the overrides."""
    for key, value in overrides.items():
        if isinstance(value, dict) and key in source and isinstance(source[key], dict):
            deep_update(source[key], value)
        else:
            source[key] = value
    return source



def initialize(config={}):

    log.info("******************************************************************")
    log.info("🧬 JANUS: Joint ANalysis for augmentation of clUSter specificity")
    log.info("******************************************************************")
    log.started("Initialization")

    result_file = ".janus_result.pkl"
    if os.path.exists(result_file):
        log.info(f"{result_file} already exists. It will be removed and recreated.")
        os.remove(result_file)  # Remove the file

    default_config = {
        "color_map": "RdYlBu",
        "output_folder": "output",
        "preprocessing": {
            "normalize": False,
            "fill_na": False,
            "drop_na": True,
        }
    }
    
    log.progress("Saving configuration settings.")   
    if config is not None:
        config = deep_update(default_config, config)
    else:
        config = default_config
        
    dsave(config, "config")
    update_matploblib_config(config)
    output_folder = config.get("output_folder", "output")
    os.makedirs(output_folder, exist_ok=True)
    log.progress(f"Output folder '{output_folder}' ensured to exist.")
    log.done("Initialization completed. ")
    tprint("JANUS",font="standard")


def update_matploblib_config(config={}):
    log.progress("Updating matplotlib settings.")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.size': 7,                # General font size
        'axes.titlesize': 10,          # Title size
        'axes.labelsize': 7,           # Axis labels (xlabel/ylabel)
        'legend.fontsize': 7,          # Legend text
        'xtick.labelsize': 6,          # X-axis tick labels
        'ytick.labelsize': 6,          # Y-axis tick labels
        'lines.linewidth': 1.5,        # Line width for plots
        'figure.dpi': 300,             # Figure resolution
        'figure.figsize': (8, 6),      # Default figure size
        'grid.linestyle': '--',        # Grid line style
        'grid.linewidth': 0.5,         # Grid line width
        'grid.alpha': 0.2,             # Grid transparency
        'axes.spines.right': False,    # Hide right spine
        'axes.spines.top': False,      # Hide top spine
        'image.cmap': config['color_map'],        # Default colormap
        'axes.edgecolor': 'black',                # Axis edge color
        'axes.facecolor': 'none',                 # Transparent axes background
        'mathtext.fontset': 'dejavusans',   # ADD THIS TO PREVENT cmsy10
        'text.usetex': False                # Ensure LaTeX is off
    })
    log.done("Matplotlib settings updated.")
