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
