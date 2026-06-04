import tensorflow as tf
from tensorflow import keras


def build_model():
    skip = keras.Input(shape=(56, 56,))  # [256, 56, 56]
    x = keras.layers.Conv2D(64, 1, strides=1, padding='valid')(skip)  # [64, 56, 56]
    x = keras.layers.BatchNormalization()(x)  # [64, 56, 56]
    x = keras.layers.ReLU()(x)  # [64, 56, 56]
    x = keras.layers.Conv2D(64, 3, strides=1, padding='same')(x)  # [64, 56, 56]
    x = keras.layers.BatchNormalization()(x)  # [64, 56, 56]
    x = keras.layers.ReLU()(x)  # [64, 56, 56]
    x = keras.layers.Conv2D(256, 1, strides=1, padding='valid')(x)  # [256, 56, 56]
    main_out = keras.layers.BatchNormalization()(x)  # [256, 56, 56]
    x = keras.layers.Conv2D(256, 1, strides=1, padding='valid')(skip)  # [256, 56, 56]
    skip_out = keras.layers.BatchNormalization()(x)  # [256, 56, 56]
    x = keras.layers.Add()([main_out, skip_out])  # [256, 56, 56]
    x = keras.layers.ReLU()(x)  # [256, 56, 56]
    # output  # [256, 56, 56]
    return keras.Model(inputs=skip, outputs=x)
