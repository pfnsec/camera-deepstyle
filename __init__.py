import sys
import os


from .src import transform

#from .src.utils import save_img, get_img, exists, list_files

import numpy as np
import pdb

import scipy.misc
import tensorflow as tf


from collections import defaultdict
import time
import json
import numpy

import batik
from prefect.core.task import Task

#from moviepy.video.io.VideoFileClip import VideoFileClip
#import moviepy.video.io.ffmpeg_writer as ffmpeg_writer

BATCH_SIZE = 4
DEVICE = '/gpu:0'



device_t = '/gpu:0'
batch_size = 1
checkpoint = "./checkpoints/wave.ckpt"

g = tf.Graph()

soft_config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
soft_config.gpu_options.allow_growth = True

with g.as_default(), g.device(device_t):

    sess = tf.compat.v1.Session(config=soft_config)
    img_placeholder = tf.compat.v1.placeholder(tf.float32, shape=[1, 692, 1280, 3],
                                        name='img_placeholder')

    preds = transform.net(img_placeholder)

    saver = tf.compat.v1.train.Saver()

    saver.restore(sess, checkpoint)

#img = get_img('laputa.png')
#print(img.shape)
#img.shape = (1, 692, 1280, 3)
#_preds = sess.run(preds, feed_dict={img_placeholder: img})
#print(_preds.shape)
#save_img("krab_out.png", _preds[0, :, :, :])

class run(Task):
    def run(self, img, **kwargs):
        img = np.array(img)
        print(img.shape)

        img.shape = (1, 692, 1280, 3)

        _preds = sess.run(preds, feed_dict={img_placeholder: img})

        #print(_preds)

        return _preds[0, :, :, :]
        


