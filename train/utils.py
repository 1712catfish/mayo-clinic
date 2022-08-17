# print(INPUT_PATH)

try:
    assert INTERACTIVE
except Exception:
    from generic_utils import *


def tf_load_image(img_path, img_shape=(512, 512, 3)):
    """ Load an image with the correct size and shape """
    img = tf.image.decode_image(tf.io.read_file(img_path), channels=img_shape[-1])
    img = tf.reshape(img, img_shape)
    return img


def augment_batch(img_batch):
    img_batch = tf.image.random_brightness(img_batch, 0.2)
    img_batch = tf.image.random_contrast(img_batch, 0.5, 2.0)
    img_batch = tf.image.random_saturation(img_batch, 0.75, 1.25)
    img_batch = tf.image.random_hue(img_batch, 0.1)
    return img_batch


def get_dataset(df, shuffle=True, buffer=512,
                batch_size=None, drop_last=False,
                cache=True, repeat=True,
                augment=True, ):
    dataset = tf.data.Dataset.from_tensor_slices((df.image_path.values, df.label.map(S2I_LBL_MAP).values))
    dataset = dataset.map(lambda x, y: (tf_load_image(x, INPUT_SHAPE), tf.one_hot(y, N_CLASSES, dtype=tf.uint8)),
                          num_parallel_calls=AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(buffer)
    if batch_size is not None:
        dataset = dataset.batch(batch_size, drop_remainder=drop_last)
    if cache:
        dataset = dataset.cache()
    if repeat:
        dataset = dataset.cache()
    if augment:
        dataset = dataset.map(lambda x, y: (augment_batch(x), y), num_parallel_calls=AUTOTUNE)
    dataset = dataset.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def build_mcsai_notile_model(tf_keras_model_fn, _weights="imagenet", top_dropout=0.5):
    _inputs = tf.keras.layers.Input(shape=INPUT_SHAPE, dtype=tf.float32)
    _bb = tf_keras_model_fn(include_top=False, input_shape=INPUT_SHAPE, weights=_weights, pooling="avg")
    x = tf.keras.layers.Dropout(top_dropout)(_bb(_inputs))
    _outputs = tf.keras.layers.Dense(N_CLASSES, activation="softmax")(x)
    return tf.keras.Model(inputs=_inputs, outputs=_outputs)


def get_class_weights(df):
    __min_count = train_df.label.value_counts().values.min()
    _class_weights = {S2I_LBL_MAP[_cls]: __min_count / _cnt for _cls, _cnt in train_df.label.value_counts().items()}
    return _class_weights


TRAIN_DIR = os.path.join(INPUT_PATH, 'train')
TRAIN_CSV = os.path.join(INPUT_PATH, 'train.csv')
print(TRAIN_CSV)

TEST_DIR = os.path.join(INPUT_PATH, 'test')
TEST_CSV = os.path.join(INPUT_PATH, 'test.csv')

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

df = pd.read_csv(TRAIN_CSV)
df["image_path"] = df["image_id"].apply(lambda x: os.path.join(IMAGE_DIR, "train", x + ".jpg"))
test_df = pd.read_csv(TEST_CSV)
test_df["image_path"] = test_df["image_id"].apply(lambda x: os.path.join(IMAGE_DIR, "test", x + ".jpg"))

train_df, val_df = k_fold_train_test_split(df.copy())
N_TRAIN, N_VAL = len(train_df), len(val_df)

train_ds = get_dataset(train_df, shuffle=True, buffer=SHUFFLE_BUFFER,
                       batch_size=BATCH_SIZE, drop_last=True,
                       cache=True, repeat=True, augment=True)
val_ds = get_dataset(val_df, shuffle=False,
                     batch_size=BATCH_SIZE, drop_last=False,
                     cache=False, repeat=False, augment=False)

class_weights = get_class_weights(train_df)
for k, v in class_weights.items():
    print(f"{k} --> {v:.4f}")
