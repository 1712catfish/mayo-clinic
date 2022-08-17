try:
    assert INTERACTIVE
except Exception:
    from dependencies import *

INPUT_PATH = '/kaggle/input/mayo-clinic-strip-ai'
IMAGE_DIR = '/kaggle/input/jpg-images-strip-ai'
GCS_PATH = 'gs://kds-dcf15753e60a6c61a37238f890ada2818c287fd8745815b311985bc8'

INPUT_SHAPE = (512, 512, 3)
S2I_LBL_MAP = {'CE': 0, 'LAA': 1}
I2S_LBL_MAP = {v: k for k, v in S2I_LBL_MAP.items()}
N_CLASSES = len(S2I_LBL_MAP)

VERBOSE = 1
