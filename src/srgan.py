"""SRGAN generator, discriminator and VGG19 feature extractor (Ledig et al., 2017)."""
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from keras.applications import VGG19
from keras.applications.vgg19 import preprocess_input as vgg19_preprocess

import config as C


@keras.saving.register_keras_serializable(package="srgan")
class PixelShuffle(layers.Layer):
    """Sub-pixel convolution upsampling: (H, W, C*r^2) -> (H*r, W*r, C)."""
    def __init__(self, scale=2, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale
    def call(self, x):
        return tf.nn.depth_to_space(x, self.scale)
    def get_config(self):
        return {**super().get_config(), "scale": self.scale}


def _residual_block(x):
    skip = x
    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.PReLU(shared_axes=[1, 2])(x)
    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    return layers.Add()([skip, x])


def _upsample_block(x):
    x = layers.Conv2D(256, 3, padding="same")(x)
    x = PixelShuffle(2)(x)
    return layers.PReLU(shared_axes=[1, 2])(x)


def build_generator():
    inp = keras.Input(shape=(C.LR_SIZE, C.LR_SIZE, 3))          # LR in [0,1]
    x = layers.Conv2D(64, 9, padding="same")(inp)
    x = skip = layers.PReLU(shared_axes=[1, 2])(x)
    for _ in range(C.N_RES_BLOCKS):
        x = _residual_block(x)
    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization(momentum=0.8)(x)
    x = layers.Add()([skip, x])
    for _ in range(int(np.log2(C.SCALE))):                        # 2 x (x2) = x4
        x = _upsample_block(x)
    out = layers.Conv2D(3, 9, padding="same", activation="tanh")(x)   # SR in [-1,1]
    return keras.Model(inp, out, name="srgan_generator")


def build_discriminator():
    inp = keras.Input(shape=(C.HR_SIZE, C.HR_SIZE, 3))          # image in [-1,1]
    x = layers.Conv2D(64, 3, padding="same")(inp)
    x = layers.LeakyReLU(0.2)(x)
    for filters, stride in [(64, 2), (128, 1), (128, 2), (256, 1), (256, 2), (512, 1), (512, 2)]:
        x = layers.Conv2D(filters, 3, strides=stride, padding="same")(x)
        x = layers.BatchNormalization(momentum=0.8)(x)
        x = layers.LeakyReLU(0.2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1024)(x)
    x = layers.LeakyReLU(0.2)(x)
    out = layers.Dense(1)(x)                                      # logit
    return keras.Model(inp, out, name="srgan_discriminator")


def build_vgg_extractor():
    vgg = VGG19(include_top=False, weights=C.PRETRAINED, input_shape=(C.HR_SIZE, C.HR_SIZE, 3))
    vgg.trainable = False
    return keras.Model(vgg.input, vgg.get_layer("block5_conv4").output, name="vgg19_54")


def vgg_features(extractor, img_m11):
    """[-1,1] RGB -> VGG19 block5_conv4 features / 12.75 (paper's VGG54 content loss scaling)."""
    return extractor(vgg19_preprocess((img_m11 + 1.0) * 127.5)) / 12.75


def sr_images(generator, lr_01, batch=64):
    """LR [0,1] -> SR [0,1] (inference mode, batched)."""
    out = [generator(lr_01[i:i + batch], training=False) for i in range(0, len(lr_01), batch)]
    return tf.clip_by_value((tf.concat(out, 0) + 1.0) / 2.0, 0.0, 1.0)


def load_generator(path):
    return keras.models.load_model(path, custom_objects={"PixelShuffle": PixelShuffle})
