# utils.py

import os
import json
import logging
from functools import wraps

import numpy as np
from PIL import Image


def setup_logging(log_file=None, level=logging.INFO, fmt=None):
    """
    Configure root logger.  
    If log_file is provided, logs also go to that file.
    """
    fmt = fmt or '%(asctime)s %(levelname)-8s %(name)s: %(message)s'
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=level, format=fmt, handlers=handlers)


def ensure_dir(path):
    """
    Ensure that the directory for `path` exists.
    If path is a file, its parent directory is created.
    """
    directory = path if os.path.isdir(path) else os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def load_config(config_path):
    """
    Load a JSON configuration file and return the dict.
    Raises FileNotFoundError or JSONDecodeError on failure.
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_image(path, as_gray=True):
    """
    Load an image from disk into a numpy array.
    - as_gray=True: returns H×W uint8 array (0–255)
    - as_gray=False: returns H×W×3 uint8 RGB array
    """
    img = Image.open(path)
    if as_gray:
        img = img.convert('L')
    else:
        img = img.convert('RGB')
    arr = np.array(img)
    return arr


def save_image(img_array, out_path):
    """
    Save a numpy array as an image.
    - If 2D, saves as grayscale.
    - If 3D with shape (...,3), saves as RGB.
    """
    ensure_dir(out_path)
    if img_array.dtype != np.uint8:
        # clip & convert floats to uint8
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    img.save(out_path)


def load_watermark_bits(path):
    """
    Load a watermark bit‐array from disk.
    Supports:
      - .npy : numpy.load
      - .bin : raw bytes, returns array of 0/1 ints
      - .txt : ASCII '0'/'1' characters, returns array of 0/1 ints
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        return np.load(path).astype(np.uint8)
    data = open(path, 'rb').read()
    if ext == '.bin':
        # each byte is 0x00 or 0x01
        bits = np.frombuffer(data, dtype=np.uint8)
    else:
        # interpret ASCII '0'/'1'
        txt = data.decode('utf-8').strip()
        bits = np.array([int(c) for c in txt if c in ('0','1')], dtype=np.uint8)
    return bits


def save_watermark_bits(bits, path, binary=True):
    """
    Save a watermark bit‐array to disk.
    - binary=True: saves raw bytes (0x00 or 0x01)
    - binary=False: saves ASCII '0'/'1' in a text file
    """
    ensure_dir(path)
    arr = np.asarray(bits, dtype=np.uint8).flatten()
    if binary:
        with open(path, 'wb') as f:
            f.write(arr.tobytes())
    else:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(''.join(str(int(x)) for x in arr))


def handle_exceptions(func):
    """
    Decorator to catch and log exceptions in a function.
    """
    logger = logging.getLogger(func.__module__)

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.exception(f"Error in {func.__name__}: {e}")
            raise
    return wrapper
