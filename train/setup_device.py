from train.setup_libraries import *

MIXED_PRECISION = False
XLA_ACCELERATE = False
AUTOGRAPH_VERBOSE = False

try:
    TPU = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Using TPU')
    tf.config.experimental_connect_to_cluster(TPU)
    tf.tpu.experimental.initialize_tpu_system(TPU)
    STRATEGY = tf.distribute.experimental.TPUStrategy(TPU)
except ValueError:
    TPU = None
    print('Using GPU/CPU')
    STRATEGY = tf.distribute.get_strategy()
    if tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

if AUTOGRAPH_VERBOSE:
    tf.autograph.set_verbosity(3, True)

if MIXED_PRECISION:
    if TPU:
        policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')
    else:
        policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
    tf.keras.mixed_precision.experimental.set_policy(policy)
    print('Mixed precision enabled')

if XLA_ACCELERATE:
    tf.config.optimizer.set_jit(True)
    print('Accelerated Linear Algebra enabled')

if TPU is not None:
    save_locally = tf.saved_model.SaveOptions(experimental_io_device='/job:localhost')
    load_locally = tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
else:
    save_locally = None
    load_locally = None