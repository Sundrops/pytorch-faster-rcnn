import tensorflow as tf
import numpy as np

a = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
print(a.value())