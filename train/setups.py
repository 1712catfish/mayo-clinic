try:
    assert INTERACTIVE
except Exception:
    from dependencies import *

INPUT_PATH = '../input/mayo-clinic-strip-ai'
IMAGE_DIR = '../input/jpg-images-strip-ai'

GCS_PATH = 'gs://kds-dcf15753e60a6c61a37238f890ada2818c287fd8745815b311985bc8'

TRAIN_DIR = os.path.join(INPUT_PATH, 'train')
TRAIN_CSV = os.path.join(INPUT_PATH, 'train.csv')
TEST_DIR = os.path.join(INPUT_PATH, 'test')
TEST_CSV = os.path.join(INPUT_PATH, 'test.csv')

INPUT_SHAPE = (512, 512, 3)
S2I_LBL_MAP = {"CE": 0, "LAA": 1}
I2S_LBL_MAP = {v: k for k, v in S2I_LBL_MAP.items()}
N_CLASSES = len(S2I_LBL_MAP)

try:
    TPU = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Using TPU.')
    tf.config.experimental_connect_to_cluster(TPU)
    tf.tpu.experimental.initialize_tpu_system(TPU)
    STRATEGY = tf.distribute.experimental.TPUStrategy(TPU)
    save_locally = tf.saved_model.SaveOptions(experimental_io_device='/job:localhost')
    load_locally = tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
    IMAGE_DIR = GCS_PATH
    BATCH_SIZE = 64 * STRATEGY.num_replicas_in_sync

except ValueError:
    TPU = None
    print('Using GPU/CPU.')
    # Yield the default distribution strategy in Tensorflow
    #   --> Works on CPU and single GPU.
    STRATEGY = tf.distribute.get_strategy()
    if tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
    save_locally = None
    load_locally = None
    BATCH_SIZE = 8

tf.config.optimizer.set_jit(True)

print(f'Batch size: {BATCH_SIZE}.')

SHUFFLE_BUFFER = BATCH_SIZE * 5
AUTOTUNE = tf.data.AUTOTUNE
VERBOSE = 1

