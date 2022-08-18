try:
    INTERACTIVE
except Exception:
    from setup_device import *

IMAGE_GCS = 'gs://kds-dcf15753e60a6c61a37238f890ada2818c287fd8745815b311985bc8'

CSV_PATH = '/kaggle/input/mayo-clinic-strip-ai'
# IMAGE_PATH = '/kaggle/input/jpg-images-strip-ai'
IMAGE_PATH = IMAGE_GCS

TRAIN_CSV = os.path.join(CSV_PATH, 'train.csv')
TEST_CSV = os.path.join(CSV_PATH, 'test.csv')

TRAIN_DIR = os.path.join(IMAGE_PATH, 'train')
TEST_DIR = os.path.join(IMAGE_PATH, 'test')

IMAGE_SHAPE = (512, 512, 3)
S2I_LBL_MAP = {'CE': 0, 'LAA': 1}
I2S_LBL_MAP = {v: k for k, v in S2I_LBL_MAP.items()}
N_CLASSES = len(S2I_LBL_MAP)
VERBOSE = 1

if TPU is not None:
    BATCH_SIZE = 32 * STRATEGY.num_replicas_in_sync
    # IMAGE_DIR = IMAGE_GCS
else:
    BATCH_SIZE = 4
    # IMAGE_DIR = INPUT_PATH

print('Batch size:', BATCH_SIZE)
print('TRAIN_DIR:', TRAIN_DIR)
print('TRAIN_CSV:', TRAIN_CSV)
