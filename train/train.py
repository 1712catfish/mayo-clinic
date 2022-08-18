from train.setup_data import *
from train.setup_train import *

history = mcsai_nt_model.fit(train_ds, validation_data=val_ds,
                             epochs=EPOCHS, callbacks=callbacks,
                             verbose=VERBOSE,
                             class_weight=class_weight)
plot_history(history, metrics=('acc', 'auc'))