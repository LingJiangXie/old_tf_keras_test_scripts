from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time
import sys
import random
import tensorflow as tf
import numpy as np
import importlib
import argparse
import facenet
import lfw
import h5py
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops

import inception_resnet_v1

batch_size=64
num_iamges=2250
epoch=10
INITIAL_LEARNING_RATE=0.01
LEARNING_RATE_DECAY_FACTOR=0.95
data_path=['/home/deep-visage/Documents/workspace/tensor_utils/tfrecords/'+name for name in os.listdir('/home/deep-visage/Documents/workspace/tensor_utils/tfrecords')]
TOWER_NAME='haha'
keep_probability=0.8
train_dir='/home/deep-visage/checkpoints/facenet_log'

def get_input():

    filename_queue = tf.train.string_input_producer(data_path,shuffle=True)
    reader =tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example, features={
        'label': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string),
    })

    imagex = tf.image.decode_image(features['image_raw'], channels=3)
    imagex.set_shape((182, 182, 3))



    imagex = tf.py_func(facenet.random_rotate_image, [imagex], tf.uint8)
    imagex = tf.random_crop(imagex, [160, 160, 3])
    imagex = tf.image.resize_image_with_crop_or_pad(imagex, 160, 160)
    image = tf.image.random_flip_left_right(imagex)

    image = tf.cast(image, tf.float32)

    label = tf.cast(features['label'], tf.int32)
    capacity = 1000 + 3 * batch_size
    '''
    min_after_dequeue=1000
    capacity =min_after_dequeue+3*batch_size

    return tf.train.shuffle_batch([image,label], batch_size=batch_size,
            capacity=capacity,min_after_dequeue=min_after_dequeue,
            allow_smaller_final_batch=True)
    '''
    return  tf.train.batch(
        [image,label],batch_size=batch_size,capacity=capacity,num_threads=3

)


def get_loss(logits, labels):

  labels = tf.cast(labels, tf.int64)

  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='cross_entropy_per_example')

  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

  tf.add_to_collection('losses', cross_entropy_mean)


  return tf.add_n(tf.get_collection('losses'), name='total_loss')

def tower_loss(scope, images, labels):

    prelogits, _ = inception_resnet_v1.inference(images, keep_probability,
                                     phase_train=True, bottleneck_layer_size=128,
                                     weight_decay=5e-5)
    logits = slim.fully_connected(prelogits, 54, activation_fn=None,
                                  weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                  weights_regularizer=slim.l2_regularizer(5e-5),
                                  scope='Logits', reuse=False)


    labels = tf.cast(labels, tf.int64)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='cross_entropy_per_example')

    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

    tf.add_to_collection('losses', cross_entropy_mean)


    #tf.add_n(tf.get_collection('losses'), name='total_loss')


    losses = tf.get_collection('losses', scope)


    total_loss = tf.add_n(losses, name='total_loss')

    return total_loss


def average_gradients(tower_grads):

  average_grads = []

  for grad_and_vars in zip(*tower_grads):

    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
    return average_grads

def train():

    with tf.Graph().as_default(),tf.device('/cpu:0'):

        image_batch, label_batch = get_input()

        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)

        # Calculate the learning rate schedule.
        num_batches_per_epoch = (num_iamges /batch_size)

        decay_steps = int(num_batches_per_epoch * epoch)

        # Decay the learning rate exponentially based on the number of steps.
        lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                        global_step,
                                        decay_steps,
                                        LEARNING_RATE_DECAY_FACTOR,
                                        staircase=True)

        # Create an optimizer that performs gradient descent.
        opt = tf.train.GradientDescentOptimizer(lr)

        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(4):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % (TOWER_NAME, i)) as scope:

                        loss = tower_loss(scope, image_batch, label_batch)

                        # Reuse variables for the next tower.
                        tf.get_variable_scope().reuse_variables()

                        # Retain the summaries from the final tower.
                        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                        # Calculate the gradients for the batch of data on this CIFAR tower.
                        grads = opt.compute_gradients(loss)

                        # Keep track of the gradients across all towers.
                        tower_grads.append(grads)


        grads = average_gradients(tower_grads)


        summaries.append(tf.summary.scalar('learning_rate', lr))



                # Apply the gradients to adjust the shared variables.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)



        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(
            0.99, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        # Group all updates to into a single train op.
        train_op = tf.group(apply_gradient_op, variables_averages_op)

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())

        # Build the summary operation from the last tower summaries.
        summary_op = tf.summary.merge(summaries)

        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()

        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=True))
        sess.run(init)

        # Start the queue runners.
        coord=tf.train.Coordinator()
        threads=tf.train.start_queue_runners(sess=sess,coord=coord)

        summary_writer = tf.summary.FileWriter(train_dir, sess.graph)

        for step in range(35):
            print('start train !')
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time

            #assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                num_examples_per_step = batch_size * 4
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = duration / 4

                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print(format_str % (datetime.now(), step, loss_value,
                                    examples_per_sec, sec_per_batch))

            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % 1000 == 0 or (step + 1) == 35:
                checkpoint_path = os.path.join(train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

            print(duration)
            
        coord.request_stop()
        coord.join(threads)


train()