try:
    INTERACTIVE
except Exception:
    from compile import *

history = mcsai_nt_model.fit(train_ds, validation_data=val_ds,
                             epochs=12, callbacks=callbacks,
                             verbose=VERBOSE,
                             class_weight=class_weights)
plot_history(history, metrics=('acc', 'auc'))