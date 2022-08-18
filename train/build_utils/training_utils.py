try:
    INTERACTIVE
except Exception:
    from train.setup_configs import *


def mcsai_notile_model(tf_keras_model_fn, _weights="imagenet", top_dropout=0.5):
    _inputs = tf.keras.layers.Input(shape=IMAGE_SHAPE, dtype=tf.float32)
    _bb = tf_keras_model_fn(include_top=False, input_shape=IMAGE_SHAPE, weights=_weights, pooling="avg")
    x = tf.keras.layers.Dropout(top_dropout)(_bb(_inputs))
    _outputs = tf.keras.layers.Dense(N_CLASSES, activation="softmax")(x)
    return tf.keras.Model(inputs=_inputs, outputs=_outputs)
