try:
    INTERACTIVE
except Exception:
    from train.setup_libraries import *


def flatten(nested_list):
    """ Flatten a list of lists """
    return [item for sublist in nested_list for item in sublist]


def k_fold_train_test_split(df, n_splits=8):
    """ Get a single stratified fold """
    gkf = GroupKFold(n_splits=n_splits)
    for train_ids, val_ids in gkf.split(df["image_id"], df["label"] + df["center_id"].astype(str), df["patient_id"]):
        # Get subsets of dataset according to split
        train_df = df.iloc[train_ids]
        val_df = df.iloc[val_ids]

        # Shuffle datasets
        train_df = train_df.sample(len(train_df)).reset_index(drop=True)
        val_df = val_df.sample(len(val_df)).reset_index(drop=True)

        return train_df, val_df


def auto_class_weight(y):
    classes = list(set(y))
    weights = class_weight.compute_class_weight('balanced', classes=classes, y=y)
    return {k: v for k, v in zip(classes, weights)}


def plot_history(history, metrics=None):
    if metrics is None:
        metrics = ['acc']
    print(history.history.keys())

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    print()

    for metric in metrics:
        plt.plot(history.history[metric])
        plt.plot(history.history[f'val_{metric}'])
        plt.title(f'model {metric}')
        plt.ylabel(metric)
        plt.xlabel('Epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()


class GarbageCollectorCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        tf.keras.backend.clear_session()


def AdamW_with_warmup(
        decay_steps,
        warmup_steps,
        learning_rate_decay_rate=0.96,
        initial_learning_rate=0.01,
        weight_decay_rate=0.0,
        staircase=False,
        name="AdamWeightDecay",
):
    return AdamWeightDecay(
        weight_decay_rate=weight_decay_rate,
        name=name,
        learning_rate=WarmUp(
            initial_learning_rate=initial_learning_rate,
            warmup_steps=warmup_steps,
            decay_schedule_fn=tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=initial_learning_rate,
                decay_steps=decay_steps,
                decay_rate=learning_rate_decay_rate,
                staircase=staircase)
        )
    )


def get_mat(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):
    # returns 3x3 transformation matrix which transforms indices

    # CONVERT DEGREES TO RADIANS
    rotation = math.pi * rotation / 180.
    shear = math.pi * shear / 180.

    # ROTATION MATRIX
    c1 = tf.math.cos(rotation)
    s1 = tf.math.sin(rotation)
    one = tf.constant([1], dtype='float32')
    zero = tf.constant([0], dtype='float32')
    rotation_matrix = tf.reshape(tf.concat([c1, s1, zero, -s1, c1, zero, zero, zero, one], axis=0), [3, 3])

    # SHEAR MATRIX
    c2 = tf.math.cos(shear)
    s2 = tf.math.sin(shear)
    shear_matrix = tf.reshape(tf.concat([one, s2, zero, zero, c2, zero, zero, zero, one], axis=0), [3, 3])

    # ZOOM MATRIX
    zoom_matrix = tf.reshape(
        tf.concat([one / height_zoom, zero, zero, zero, one / width_zoom, zero, zero, zero, one], axis=0), [3, 3])

    # SHIFT MATRIX
    shift_matrix = tf.reshape(tf.concat([one, zero, height_shift, zero, one, width_shift, zero, zero, one], axis=0),
                              [3, 3])

    return tf.keras.backend.dot(tf.keras.backend.dot(rotation_matrix, shear_matrix),
                                tf.keras.backend.dot(zoom_matrix, shift_matrix))
