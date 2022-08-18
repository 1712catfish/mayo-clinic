try:
    INTERACTIVE
except Exception:
    from build_utils.training_utils import *
    from setup_data import *

EPOCHS = 12

with STRATEGY.scope():
    OPTIMIZER = AdamW_with_warmup(decay_steps=STEPS_PER_EPOCH,
                                  warmup_steps=4 * STEPS_PER_EPOCH,
                                  learning_rate_decay_rate=0.75,
                                  initial_learning_rate=1e-5,
                                  weight_decay_rate=0.0, )
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
    mcsai_nt_model = mcsai_notile_model(tf.keras.applications.EfficientNetB6)
    mcsai_nt_model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=METRICS)

mcsai_nt_model.summary()
