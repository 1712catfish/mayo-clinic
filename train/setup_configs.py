try:
    INTERACTIVE
except Exception:
    from setup_device import *

INPUT_PATH = Path('/kaggle/input/mayo-clinic-strip-ai')
IMAGE_DIR = Path('/kaggle/input/jpg-images-strip-ai')

CSV_GCS = 'gs://kds-5b35e8caec4c36d580aef4332f9e7811adbc4591990a3a47a73c3fe0'
IMAGE_GCS = 'gs://kds-dcf15753e60a6c61a37238f890ada2818c287fd8745815b311985bc8'

TRAIN_DIR = INPUT_PATH / 'train'
TRAIN_CSV = INPUT_PATH / 'train.csv'
TEST_DIR = INPUT_PATH / 'test'
TEST_CSV = INPUT_PATH / 'test.csv'

IMAGE_SHAPE = (512, 512, 3)
S2I_LBL_MAP = {'CE': 0, 'LAA': 1}
I2S_LBL_MAP = {v: k for k, v in S2I_LBL_MAP.items()}
N_CLASSES = len(S2I_LBL_MAP)
VERBOSE = 1

if TPU is not None:
    BATCH_SIZE = 32 * STRATEGY.num_replicas_in_sync
    IMAGE_DIR = IMAGE_GCS
else:
    BATCH_SIZE = 4
    IMAGE_DIR = INPUT_PATH

print('Batch size:', BATCH_SIZE)
print('IMAGE_DIR:', IMAGE_DIR)
