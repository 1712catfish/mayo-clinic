try:
    assert INTERACTIVE
except Exception:
    from dependencies import *

try:
    TPU = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Using TPU')
    tf.config.experimental_connect_to_cluster(TPU)
    tf.tpu.experimental.initialize_tpu_system(TPU)
    STRATEGY = tf.distribute.experimental.TPUStrategy(TPU)
    BATCH_SIZE = 64 * STRATEGY.num_replicas_in_sync
except ValueError:
    TPU = None
    print('Using GPU/CPU')
    # Yield the default distribution strategy in Tensorflow
    #   --> Works on CPU and single GPU.
    STRATEGY = tf.distribute.get_strategy()
    if tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
    BATCH_SIZE = 8

GCS_PATH = 'gs://kds-dcf15753e60a6c61a37238f890ada2818c287fd8745815b311985bc8'