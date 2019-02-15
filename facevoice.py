#!/usr/bin/env python
#
# This file contains a reference implementation of the following paper:
#   On Learning Associations of Faces and Voices
#   Changil Kim, Hijung Valentina Shin, Tae-Hyun Oh, Alexandre Kaspar, Mohamed Elgharib, Wojciech Matusik
#   ACCV 2018
# Please cite the above paper if you use this software.
# More information including a pre-trained model can be found in the project website:
#   http://facevoice.csail.mit.edu/
#
# Usage:
#   voice-to-face:  facevoice.py v2f -c <checkpoint> --voice <voice-file> --face0 <face-file> --face1 <face-file>
#   face-to-voice:  facevoice.py f2v -c <checkpoint> --face <face-file> --voice0 <voice-file> --voice1 <voice-file>
#
# Author: Changil Kim <changil@csail.mit.edu>
#

from __future__ import unicode_literals
from functools import reduce
import tensorflow as tf
import numpy as np
import warnings
import argparse
import skimage.io
import skimage.transform
import skimage
import scipy.io.wavfile

def conv(x, kh, kw, co, sh, sw, name, relu=True, padding='SAME'):
    with tf.variable_scope(name):
        ci = x.get_shape()[-1]
        weights = tf.get_variable('weights', shape=[kh, kw, ci, co])
        biases = tf.get_variable('biases', shape=[co])
        x = tf.nn.conv2d(x, weights, [1, sh, sw, 1], padding=padding)
        x = tf.nn.bias_add(x, biases)
        if relu:
            x = tf.nn.relu(x)
        return x

def fc(x, co, name, relu=True):
    with tf.variable_scope(name):
        # flatten the input tensor
        ci = reduce(lambda a, b: a*b, x.get_shape()[1:].as_list())
        x = tf.reshape(x, [-1, ci])
        weights = tf.get_variable('weights', shape=[ci, co])
        biases = tf.get_variable('biases', shape=[co])
        if relu:
            x = tf.nn.relu_layer(x, weights, biases)
        else:
            x = tf.nn.xw_plus_b(x, weights, biases)
        return x
        
def bn(x, name, relu=True):
    # for inference only
    with tf.variable_scope(name):
        ci = x.get_shape()[-1]
        scale = tf.get_variable('scale', shape=[ci])
        offset = tf.get_variable('offset', shape=[ci])
        mean = tf.get_variable('mean', shape=[ci])
        variance = tf.get_variable('variance', shape=[ci])
        x = tf.nn.batch_normalization(x, mean=mean, variance=variance, offset=offset, scale=scale, variance_epsilon=1e-5, name=name)
        if relu:
            x = tf.nn.relu(x)
        return x

def maxpool(x, kh, kw, sh, sw, name, padding='SAME'):
    return tf.nn.max_pool(x, ksize=[1, kh, kw, 1], strides=[1, sh, sw, 1], padding=padding, name=name)

def avgpool(x, kh, kw, sh, sw, name, padding='SAME'):
    return tf.nn.avg_pool(x, ksize=[1, kh, kw, 1], strides=[1, sh, sw, 1], padding=padding, name=name)

def padding(x, h, w, name):
    # for torch-style padding
    pads = np.zeros([4, 2], dtype=np.int32)
    pads[1, :] = h
    pads[2, :] = w
    return tf.pad(x, tf.constant(pads), name=name)

def facenet(x):
    # vgg16 up to conv5_3
    x = conv(x, 3, 3, 64, 1, 1, name='conv1_1')
    x = conv(x, 3, 3, 64, 1, 1, name='conv1_2')
    x = maxpool(x, 2, 2, 2, 2, name='pool1')
    x = conv(x, 3, 3, 128, 1, 1, name='conv2_1')
    x = conv(x, 3, 3, 128, 1, 1, name='conv2_2')
    x = maxpool(x, 2, 2, 2, 2, name='pool2')
    x = conv(x, 3, 3, 256, 1, 1, name='conv3_1')
    x = conv(x, 3, 3, 256, 1, 1, name='conv3_2')
    x = conv(x, 3, 3, 256, 1, 1, name='conv3_3')
    x = maxpool(x, 2, 2, 2, 2, name='pool3')
    x = conv(x, 3, 3, 512, 1, 1, name='conv4_1')
    x = conv(x, 3, 3, 512, 1, 1, name='conv4_2')
    x = conv(x, 3, 3, 512, 1, 1, name='conv4_3')
    x = maxpool(x, 2, 2, 2, 2, name='pool4')
    x = conv(x, 3, 3, 512, 1, 1, name='conv5_1')
    x = conv(x, 3, 3, 512, 1, 1, name='conv5_2')
    x = conv(x, 3, 3, 512, 1, 1, name='conv5_3')
    # global pooling
    x = avgpool(x, 14, 14, 1, 1, padding='VALID', name='globalpool')
    # fc layers
    x = fc(x, 128, name='fc1')
    x = fc(x, 128, relu=False, name='fc2')
    return x

def voicenet(x):
    # soundnet up to conv6
    x = padding(x, 32, 0, name='pad1')
    x = conv(x, 64, 1, 16, 2, 1, padding='VALID', relu=False, name='conv1')
    x = bn(x, relu=True, name='bn1')
    x = maxpool(x, 8, 1, 8, 1, padding='VALID', name='pool1')
    x = padding(x, 16, 0, name='pad2')
    x = conv(x, 32, 1, 32, 2, 1, padding='VALID', relu=False, name='conv2')
    x = bn(x, relu=True, name='bn2')
    x = maxpool(x, 8, 1, 8, 1, padding='VALID', name='pool2')
    x = padding(x, 8, 0, name='pad3')
    x = conv(x, 16, 1, 64, 2, 1, padding='VALID', relu=False, name='conv3')
    x = bn(x, relu=True, name='bn3')
    x = padding(x, 4, 0, name='pad4')
    x = conv(x, 8, 1, 128, 2, 1, padding='VALID', relu=False, name='conv4')
    x = bn(x, relu=True, name='bn4')
    x = padding(x, 2, 0, name='pad5')
    x = conv(x, 4, 1, 256, 2, 1, padding='VALID', relu=False, name='conv5')
    x = bn(x, relu=True, name='bn5')
    x = maxpool(x, 4, 1, 4, 1, padding='VALID', name='pool5')
    x = padding(x, 2, 0, name='pad6')
    x = conv(x, 4, 1, 512, 2, 1, padding='VALID', relu=False, name='conv6')
    # global pooling
    x = avgpool(x, 14, 1, 1, 1, padding='VALID', name='globalpool')
    # fc layers
    x = fc(x, 128, name='fc1')
    x = fc(x, 128, relu=False, name='fc2')
    return x

def voice2face(voice, face0, face1):
    with tf.variable_scope('voice'):
        voicerep = voicenet(voice)
    with tf.variable_scope('face'):
        facerep0 = facenet(face0)
    with tf.variable_scope('face', reuse=True):
        facerep1 = facenet(face1)

    # pick the closer face
    return which(voicerep, facerep0, facerep1)

def face2voice(face, voice0, voice1):
    with tf.variable_scope('face'):
        facerep = facenet(face)
    with tf.variable_scope('voice'):
        voicerep0 = voicenet(voice0)
    with tf.variable_scope('voice', reuse=True):
        voicerep1 = voicenet(voice1)

    # pick the closer voice
    return which(facerep, voicerep0, voicerep1)

def which(repref, rep0, rep1):
    # pick the one closer to repref between rep0 and rep1
    dist_sq = lambda a, b: tf.reduce_sum(tf.square(a - b), axis=1)
    d0 = dist_sq(repref, rep0)
    d1 = dist_sq(repref, rep1)
    batch_size = tf.shape(d0)[0]
    verdict = tf.where(d0 < d1, tf.fill(dims=[batch_size], value=0), tf.fill(dims=[batch_size], value=1))
    return verdict

def load_wave(wave_file):
    rate, data = scipy.io.wavfile.read(wave_file)
    if rate != 22050:
        raise RuntimeError('input wav must be sampled at 22,050 Hz')
    if data.ndim > 1:
        # take the left channel
        data = data[:, 0]
    if data.shape[0] < 220500:
        # make the wav at least 10-second long
        data = np.tile(data, (220500 + data.shape[0] - 1) // data.shape[0])
    # take the first 10 seconds
    data = np.reshape(data[:220500], [-1, 1, 1]).astype(np.float32)
    return data

def load_image(image_file):
    image = skimage.io.imread(image_file)
    if image.ndim != 3 or image.shape[2] != 3:
        raise RuntimeError('input image must be in color')
    # resize
    image = skimage.transform.resize(image, [256, 256], mode='reflect', anti_aliasing=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        image = skimage.img_as_ubyte(image)
    # crop
    image = image[16:-16, 16:-16]
    # subtract the mean
    image = image.astype(np.float32) - np.array([124., 117., 104.], dtype=np.float32)
    # RGB to BGR
    image = image[:, :, ::-1]
    return image
    
def testv2f(opt):
    # load files
    voice = np.expand_dims(load_wave(opt.voice), axis=0)
    face0 = np.expand_dims(load_image(opt.face0), axis=0)
    face1 = np.expand_dims(load_image(opt.face1), axis=0)

    # create placeholders for the input
    voicenode = tf.placeholder(tf.float32, shape=(None, 220500, 1, 1), name='voicenode')
    facenode0 = tf.placeholder(tf.float32, shape=(None, 224, 224, 3), name='facenode0')
    facenode1 = tf.placeholder(tf.float32, shape=(None, 224, 224, 3), name='facenode1')

    # set up network
    verdictnode = voice2face(voicenode, facenode0, facenode1)

    saver = tf.train.Saver(save_relative_paths=True)

    with tf.Session() as sess:
        # load checkpoint
        saver.restore(sess, tf.train.latest_checkpoint(opt.checkpoint_dir))

        # test
        verdict = sess.run(verdictnode, feed_dict={voicenode: voice, facenode0: face0, facenode1: face1})

        print('predicted matching face: face{} ({})'.format(verdict[0], [opt.face0, opt.face1][verdict[0]]))

def testf2v(opt):
    # load files
    face = np.expand_dims(load_image(opt.face), axis=0)
    voice0 = np.expand_dims(load_wave(opt.voice0), axis=0)
    voice1 = np.expand_dims(load_wave(opt.voice1), axis=0)

    # create placeholders for the input
    facenode = tf.placeholder(tf.float32, shape=(None, 224, 224, 3), name='facenode')
    voicenode0 = tf.placeholder(tf.float32, shape=(None, 220500, 1, 1), name='voicenode0')
    voicenode1 = tf.placeholder(tf.float32, shape=(None, 220500, 1, 1), name='voicenode1')

    # set up network
    verdictnode = face2voice(facenode, voicenode0, voicenode1)

    saver = tf.train.Saver(save_relative_paths=True)

    with tf.Session() as sess:
        # load checkpoint
        saver.restore(sess, tf.train.latest_checkpoint(opt.checkpoint_dir))

        # test
        verdict = sess.run(verdictnode, feed_dict={facenode: face, voicenode0: voice0, voicenode1: voice1})

        print('predicted matching voice: voice{} ({})'.format(verdict[0], [opt.voice0, opt.voice1][verdict[0]]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    # v2f
    parserv2f = subparsers.add_parser('v2f')
    parserv2f.add_argument('-c', '--checkpoint-dir', required=True)
    parserv2f.add_argument('--voice', required=True)
    parserv2f.add_argument('--face0', required=True)
    parserv2f.add_argument('--face1', required=True)
    parserv2f.set_defaults(func=testv2f)
    # f2v
    parserf2v = subparsers.add_parser('f2v')
    parserf2v.add_argument('-c', '--checkpoint-dir', required=True)
    parserf2v.add_argument('--face', required=True)
    parserf2v.add_argument('--voice0', required=True)
    parserf2v.add_argument('--voice1', required=True)
    parserf2v.set_defaults(func=testf2v)

    opt = parser.parse_args()
    opt.func(opt)

