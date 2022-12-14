# Machine Learning and Data Science Imports
import tensorflow as tf
import tensorflow_hub as tfhub
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
import pandas as pd

pd.options.mode.chained_assignment = None
import numpy as np
import sklearn
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from pandarallel import pandarallel

pandarallel.initialize()
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.utils import class_weight
from scipy.spatial import cKDTree

# # RAPIDS
# import cudf, cupy, cuml
# from cuml.neighbors import NearestNeighbors
# from cuml.manifold import TSNE, UMAP
# from cuml import PCA

# Built In Imports
# from kaggle_datasets import KaggleDatasets
from collections import Counter
from datetime import datetime
from zipfile import ZipFile
from glob import glob
import openslide
import warnings
import requests
import hashlib
import imageio
import IPython
import sklearn
import urllib
import zipfile
import pickle
import random
import shutil
import string
import json
import math
import time
import gzip
import ast
import sys
import io
import os
import gc
import re
from pathlib import Path
from transformers.optimization_tf import WarmUp, AdamWeightDecay

# Visualization Imports
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
# import plotly.graph_objects as go
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

tqdm.pandas()
# import plotly.express as px
import tifffile as tif
import seaborn as sns
from PIL import Image, ImageEnhance

Image.MAX_IMAGE_PIXELS = 5_000_000_000
import matplotlib

from matplotlib import animation, rc

rc('animation', html='jshtml')
# import plotly
import PIL
import cv2

# import plotly.io as pio
# print(pio.renderers)

def seed_it_all(seed=7):
    """ Attempt to be Reproducible """
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = "1"
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


seed_it_all()