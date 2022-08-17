try:
    assert INTERACTIVE
except Exception:
    from utils import *

with STRATEGY.scope():
    OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=0.01)
    LOSS = 'categorical_crossentropy'
    METRICS = ['acc', tf.keras.metrics.AUC(name='auc')]
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=4, verbose=VERBOSE,
                                         mode='max', restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint('./efficientnetb6_512_notile', monitor='val_auc', mode='max',
                                           save_best_only=True, options=save_locally),
        # tf.keras.callbacks.LearningRateScheduler(lambda epoch: lr_function(epoch)),
        # GarbageCollectorCallback(),
    ]
    mcsai_nt_model = build_mcsai_notile_model(tf.keras.applications.EfficientNetB6)
    mcsai_nt_model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=METRICS)

mcsai_nt_model.summary()
