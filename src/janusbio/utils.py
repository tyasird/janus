import os
import joblib
from art import tprint

from .logging_config import log


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
            return category_data.get(name, {})
        return category_data
    return {}


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
        os.remove(result_file)

    default_config = {
        "color_map": "RdYlBu",
        "output_folder": "output",
        "preprocessing": {
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
    tprint("JANUS", font="standard")


def update_matploblib_config(config={}):
    log.progress("Updating matplotlib settings.")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.size': 7,
        'axes.titlesize': 10,
        'axes.labelsize': 7,
        'legend.fontsize': 7,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'lines.linewidth': 1.5,
        'figure.dpi': 300,
        'figure.figsize': (8, 6),
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
        'grid.alpha': 0.2,
        'axes.spines.right': False,
        'axes.spines.top': False,
        'image.cmap': config['color_map'],
        'axes.edgecolor': 'black',
        'axes.facecolor': 'none',
        'mathtext.fontset': 'dejavusans',
        'text.usetex': False
    })
    log.done("Matplotlib settings updated.")