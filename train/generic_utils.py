try:
    assert INTERACTIVE
except Exception:
    from setups import *


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


class GarbageCollectorCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        tf.keras.backend.clear_session()


def plot_history(_history, fold_num="1", metrics="acc"):
    """ TBD """
    fig = px.line(_history.history,
                  x=range(len(_history.history["loss"])),
                  y=["loss", "val_loss"],
                  labels={"value": "Loss (log-axis)", "x": "Epoch #"},
                  title=f"<b>FOLD {fold_num} MODEL - LOSS</b>", log_y=True
                  )
    fig.show()

    for _m in metrics:
        fig = px.line(_history.history,
                      x=range(len(_history.history[_m])),
                      y=[_m, f"val_{_m}"],
                      labels={"value": f"{_m} (log-axis)", "x": "Epoch #"},
                      title=f"<b>FOLD {fold_num} MODEL - {_m}</b>", log_y=True)
        fig.show()
