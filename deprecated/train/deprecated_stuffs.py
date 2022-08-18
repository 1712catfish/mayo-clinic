# def auto_class_weight(dataframe):
#     # TODO: Use sklearn class_weight
#     __min_count = dataframe.label.value_counts().values.min()
#     _class_weights = {S2I_LBL_MAP[_cls]: __min_count / _cnt for _cls, _cnt in dataframe.label.value_counts().items()}
#     return _class_weights

# def tf_load_image(img_path, img_shape=(512, 512, 3)):
#     """ Load an image with the correct size and shape """
#     img = tf.image.decode_image(tf.io.read_file(img_path), channels=img_shape[-1])
#     img = tf.reshape(img, img_shape)
#     return img

# def augment(img_batch, label):
#     img_batch = tf.image.random_brightness(img_batch, 0.2)
#     img_batch = tf.image.random_contrast(img_batch, 0.5, 2.0)
#     img_batch = tf.image.random_saturation(img_batch, 0.75, 1.25)
#     img_batch = tf.image.random_hue(img_batch, 0.1)
#     return img_batch, label

# def plot_history(_history, fold_num="1", metrics="acc"):
#     """ TBD """
#     fig = px.line(_history.history,
#                   x=range(len(_history.history["loss"])),
#                   y=["loss", "val_loss"],
#                   labels={"value": "Loss (log-axis)", "x": "Epoch #"},
#                   title=f"<b>FOLD {fold_num} MODEL - LOSS</b>", log_y=True)
#     fig.show()
#
#     for _m in metrics:
#         fig = px.line(_history.history,
#                       x=range(len(_history.history[_m])),
#                       y=[_m, f"val_{_m}"],
#                       labels={"value": f"{_m} (log-axis)", "x": "Epoch #"},
#                       title=f"<b>FOLD {fold_num} MODEL - {_m}</b>", log_y=True)
#         fig.show()
