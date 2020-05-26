"""
test-gpu.py
======================================

Test if paperspace environment can use GPU in the docker image
"""
import tensorflow as tf
from tensorflow.python.client import device_lib

# print available devices
print(device_lib.list_local_devices())

# number of GPU's available
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# print tensorflow placement - need to create a small model and run to log this
tf.debugging.set_log_device_placement(True)

