import pytest
import tensorflow as tf


def test_build_model():
    model = build_model()
    x = tf.random.normal((1, 512, 64,))
    x2 = tf.random.normal((1, 512, 64,))
    x3 = tf.random.normal((1, 512, 64,))
    output = model(=, =, =, training=False)
    assert output.shape == (1, 512, 64,)
