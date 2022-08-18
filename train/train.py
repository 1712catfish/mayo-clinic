try:
    INTERACTIVE
except Exception:
    from setup_train import *

history = mcsai_nt_model.fit(train_ds,
                             steps_per_epoch=STEPS_PER_EPOCH,
                             validation_data=val_ds,
                             # validation_steps=VALIDATION_STEPS,
                             epochs=EPOCHS,
                             callbacks=callbacks,
                             verbose=VERBOSE,
                             class_weight=CLASS_WEIGHT)
plot_history(history, metrics=('acc', 'auc'))