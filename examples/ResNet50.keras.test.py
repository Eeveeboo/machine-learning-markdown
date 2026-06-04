import pytest
import tensorflow as tf


def test_build_model():
    model = build_model()
    x = tf.random.normal((1, 256, 56, 56,))
    output = model(=, training=False)
    assert output.shape == (1, 256, 56, 56,)
