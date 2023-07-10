from pathlib import Path
import re
import numpy as np
import dill
from tensorflow.keras.models import load_model
from src.file_management.directories import MODELS_DIR


# Nicely formatted time string
def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return f"{h}:{m:>02}:{s:>05.2f}"


def generate_output_dir(run_desc="", outdir_path=MODELS_DIR):
    prev_run_dirs = []
    if outdir_path.is_dir():
        prev_run_dirs = [x for x in outdir_path.iterdir() if x.is_dir()]
    prev_run_ids = [re.match(r'^\d+', x.name) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    run_dir = outdir_path / f'{cur_run_id:05d}-{run_desc}'
    assert not run_dir.exists()
    run_dir.mkdir()

    print(f"Output directory: {run_dir}")
    return Path(run_dir)


def get_latest_hdf5_pickle(run_dir: Path):

    # Define the prefix to search for
    prefix = "model-"

    # Define the path to the checkpoints directory
    checkpoints_dir = run_dir / "checkpoints"

    # Find all files in the directory that match the prefix and have a number at the end
    pattern = re.compile(f"^{prefix}\d+")
    matching_files = [f for f in checkpoints_dir.iterdir() if pattern.match(f.name)]

    # Find the latest .hdf5 file
    latest_hdf5 = max([f for f in matching_files if f.suffix == ".hdf5"], key=lambda f: float(f.name.split("-")[1].split(".")[0]))

    # Find the latest .pickle file
    latest_pickle = max([f for f in matching_files if f.suffix == ".pickle"], key=lambda f: float(f.name.split("-")[1].split(".")[0]))

    print(f"Latest HDF5 file: {latest_hdf5}")
    print(f"Latest pickle file: {latest_pickle}")

    return latest_hdf5, latest_pickle


def load_model_data(model_path: Path, opt_path: Path):
    model = load_model(model_path)
    with open(opt_path, 'rb') as fp:
        d = dill.load(fp)
        epoch = d['epoch']
        opt = d['opt']
    return epoch, model, opt


def load_latest_model(run_dir:Path):
    model_path, opt_path = get_latest_hdf5_pickle(run_dir)
    model = load_model(model_path)
    with open(opt_path, 'rb') as fp:
        d = dill.load(fp)
        epoch = d['epoch']
        opt = d['opt']
    return epoch, model, opt