try:
    INTERACTIVE
except Exception:
    from generic_utils import *
    from train.setup_configs import *


def data_augment(image, label):
    p_spatial = tf.random.uniform([], 0, 1., dtype=tf.float32)
    p_rotate = tf.random.uniform([], 0, 1., dtype=tf.float32)
    p_pixel = tf.random.uniform([], 0, 1., dtype=tf.float32)
    p_shear = tf.random.uniform([], 0, 1., dtype=tf.float32)
    p_crop = tf.random.uniform([], 0, 1., dtype=tf.float32)

    # image = tf.image.random_flip_up_down(image)
    image = tf.image.random_flip_left_right(image)

    if p_spatial > .75:
        image = tf.image.transpose(image)

    if p_pixel >= .2:
        if p_pixel >= .8:
            image = tf.image.random_saturation(image, lower=.7, upper=1.3)
        elif p_pixel >= .6:
            image = tf.image.random_contrast(image, lower=.8, upper=1.2)
        elif p_pixel >= .4:
            image = tf.image.random_brightness(image, max_delta=.1)
        else:
            image = tf.image.adjust_gamma(image, gamma=.6)

    if p_crop > .7:
        if p_crop > .9:
            image = tf.image.central_crop(image, central_fraction=.6)
        elif p_crop > .8:
            image = tf.image.central_crop(image, central_fraction=.7)
        else:
            image = tf.image.central_crop(image, central_fraction=.8)
    elif p_crop > .4:
        crop_size = tf.random.uniform([], int(IMAGE_SHAPE[0] * .6), IMAGE_SHAPE[0], dtype=tf.int32)
        image = tf.image.random_crop(image, size=[crop_size, crop_size, IMAGE_SHAPE[2]])

    image = tf.image.resize(image, size=IMAGE_SHAPE[:2])
    image = tf.reshape(image, IMAGE_SHAPE)

    return image, label


def rot_shr_z_shf(image, label):
    # input image - is one image of size [dim, dim, 3] not a batch of [b, dim, dim, 3]
    # output - image randomly rotated, sheared, zoomed, and shifted
    DIM = IMAGE_SHAPE[0]
    XDIM = DIM % 2  # fix for size 331

    rot = 15. * tf.random.normal([1], dtype='float32')
    shr = 5. * tf.random.normal([1], dtype='float32')
    h_zoom = 1.0 + tf.random.normal([1], dtype='float32') / 10.
    w_zoom = 1.0 + tf.random.normal([1], dtype='float32') / 10.
    h_shift = 16. * tf.random.normal([1], dtype='float32')
    w_shift = 16. * tf.random.normal([1], dtype='float32')

    # GET TRANSFORMATION MATRIX
    m = get_mat(rot, shr, h_zoom, w_zoom, h_shift, w_shift)

    # LIST DESTINATION PIXEL INDICES
    x = tf.repeat(tf.range(DIM // 2, -DIM // 2, -1), DIM)
    y = tf.tile(tf.range(-DIM // 2, DIM // 2), [DIM])
    z = tf.ones([DIM * DIM], dtype='int32')
    idx = tf.stack([x, y, z])

    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS
    idx2 = tf.keras.backend.dot(m, tf.cast(idx, dtype='float32'))
    idx2 = tf.keras.backend.cast(idx2, dtype='int32')
    idx2 = tf.keras.backend.clip(idx2, -DIM // 2 + XDIM + 1, DIM // 2)

    # FIND ORIGIN PIXEL VALUES
    idx3 = tf.stack([DIM // 2 - idx2[0,], DIM // 2 - 1 + idx2[1,]])
    d = tf.gather_nd(image, tf.transpose(idx3))

    return tf.reshape(d, [DIM, DIM, 3]), label
