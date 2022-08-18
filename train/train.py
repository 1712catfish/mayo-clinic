try:
    INTERACTIVE
except Exception:
    from setup_train import *

history = mcsai_nt_model.fit(train_ds, validation_data=val_ds,
                             epochs=EPOCHS, callbacks=callbacks,
                             verbose=VERBOSE,
                             class_weight=CLASS_WEIGHT)
plot_history(history, metrics=('acc', 'auc'))