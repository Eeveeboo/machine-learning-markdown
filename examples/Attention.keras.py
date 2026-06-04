import tensorflow as tf
from tensorflow import keras


def build_model():
    query = keras.Input(shape=(64,))  # [512, 64]
    key = keras.Input(shape=(64,))  # [512, 64]
    value = keras.Input(shape=(64,))  # [512, 64]
    q_proj = keras.layers.Dense(64)(query)  # [512, 64]
    k_proj = keras.layers.Dense(64)(key)  # [512, 64]
    v_proj = keras.layers.Dense(64)(value)  # [512, 64]
    x = keras.layers.Dot(axes=-1)([q_proj, k_proj])  # [512, 512]
    attn_weights = keras.layers.Softmax(axis=-1)(x)  # [512, 512]
    x = keras.layers.Dot(axes=-1)([attn_weights, v_proj])  # [512, 64]
    x = keras.layers.Dense(64)(x)  # [512, 64]
    # output  # [512, 64]
    return keras.Model(inputs=value, outputs=x)
