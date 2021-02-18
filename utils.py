import tensorflow as tf
import numpy as np
import math
import os


# read images
def read_image(img_path, size=(256,256)):
    if size is None:
        img = tf.keras.preprocessing.image.load_img(img_path)
    else:
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
    img = tf.keras.preprocessing.image.img_to_array(img , dtype='float32')
    img = img / 255.0
    return img

# image augmentation function
def data_augmentation(image, mode):
    if mode == 0:
        # original
        return image
    elif mode == 1:
        # flip up and down
        return np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        return np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        image = np.rot90(image)
        return np.flipud(image)
    elif mode == 4:
        # rotate 180 degree
        return np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        image = np.rot90(image, k=2)
        return np.flipud(image)
    elif mode == 6:
        # rotate 270 degree
        return np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        image = np.rot90(image, k=3)
        return np.flipud(image)

# correlation alignment
def coral(src, dst):
    src_flat = src.reshape(-1, 3)
    src_flat_mean = np.mean(src_flat, 0, keepdims=True)
    src_flat_std = np.std(src_flat, 0, keepdims=True)
    src_flat_norm = (src_flat - src_flat_mean) / src_flat_std
    src_flat_cov_eye = np.matmul(src_flat_norm.T, src_flat_norm) + np.eye(3)

    dst_flat = dst.reshape(-1, 3)
    dst_flat_mean = np.mean(dst_flat, 0, keepdims=True)
    dst_flat_std = np.std(dst_flat, 0, keepdims=True)
    dst_flat_norm = (dst_flat - dst_flat_mean) / dst_flat_std
    dst_flat_cov_eye = np.matmul(dst_flat_norm.T, dst_flat_norm) + np.eye(3)

    src_flat_norm_transfer = np.matmul(src_flat_norm, np.matmul(
        np.linalg.inv(_mat_sqrt(src_flat_cov_eye)),
        _mat_sqrt(dst_flat_cov_eye)
    ))
    src_flat_transfer = src_flat_norm_transfer * dst_flat_std + dst_flat_mean
    return src_flat_transfer.reshape(src.shape)


# return name from path
def get_name(path):
    return (path.split(os.sep)[-1]).split(".")[0]


def _mat_sqrt(m):
    u, s, v = np.linalg.svd(m)
    return np.matmul(np.matmul(u, np.diag(np.sqrt(s))), v)


def lerp(a, b, l):
    return (1 - l) * a + l * b


def tanh01(x):
    return tf.tanh(x) * 0.5 + 0.5


def tanh_range(l, r, initial=None):
    def get_activation(left, right, initial):
        def activation(x):
            if initial is not None: bias = math.atanh(2 * (initial - left) / (right - left) - 1)
            else: bias = 0
            return tanh01(x + bias) * (right - left) + left
        return activation
    return get_activation(l, r, initial)


def rgb2lum(image):
    image = 0.27 * image[:, :, :, 0] + 0.67 * image[:, :, :, 1] + 0.06 * image[:, :, :, 2]
    return image[:, :, :, None]


def _gaussian_kernel(kernel_size, sigma, n_channels, dtype):
    x = tf.range(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=dtype)
    g = tf.math.exp(-(tf.pow(x, 2) / (2 * tf.pow(tf.cast(sigma, dtype), 2))))
    g_norm2d = tf.pow(tf.reduce_sum(g), 2)
    g_kernel = tf.tensordot(g, g, axes=0) / g_norm2d
    g_kernel = tf.expand_dims(g_kernel, axis=-1)
    return tf.expand_dims(tf.tile(g_kernel, (1, 1, n_channels)), axis=-1)
