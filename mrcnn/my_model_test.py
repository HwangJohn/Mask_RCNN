import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import random
import datetime
import re
import math
import logging
from collections import OrderedDict
import multiprocessing
import numpy as np
import tensorflow as tf
import tensorflow.keras
import tensorflow.keras.backend as K
import tensorflow.keras.layers as KL
# import tensorflow.keras.engine as KE
import tensorflow.keras.models as KM
import utils

print(os.getcwd())

import sys
sys.path.append('samples/coco/')
sys.path.append('coco/PythonAPI')
sys.path.append('.')
import coco

from mrcnn.my_model import conv_block
from mrcnn.my_model import identity_block
from mrcnn.my_model import BatchNorm

input_image = np.zeros((10, 100, 100, 3))
input_image = tf.cast(input_image, tf.float32)
architecture = "resnet50"
stage5=False
train_bn=True

# identity_block test

x = KL.ZeroPadding2D((3, 3))(input_image)
x = KL.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x)
x = BatchNorm(name='bn_conv1')(x, training=train_bn)
x = KL.Activation('relu')(x)
C1 = x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
# Stage 2
x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), train_bn=train_bn)
x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', train_bn=train_bn)
C2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', train_bn=train_bn)
# Stage 3
x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', train_bn=train_bn)
x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', train_bn=train_bn)
x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', train_bn=train_bn)
C3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', train_bn=train_bn)
# Stage 4
x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', train_bn=train_bn)
block_count = {"resnet50": 5, "resnet101": 22}[architecture]
for i in range(block_count):
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i), train_bn=train_bn)
C4 = x
# Stage 5
if stage5:
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', train_bn=train_bn)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', train_bn=train_bn)
    C5 = x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', train_bn=train_bn)
else:
    C5 = None
[C1, C2, C3, C4, C5]