import tensorflow as tf
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import cv2

files=['/home/deep-visage/Documents/workspace/tensor_utils/tfrecords/'+name for name in os.listdir('/home/deep-visage/Documents/workspace/tensor_utils/tfrecords')]
print(files)
#files=['mini_test-00000.tfrecords','mini_test-00001.tfrecords','mini_test-00002.tfrecords','mini_test-00003.tfrecords','mini_test-00004.tfrecords','mini_test-00005.tfrecords','mini_test-00006.tfrecords']
#files=tf.train.match_filenames_once('*.tfrecords')

filename_queue=tf.train.string_input_producer(files,shuffle=True)

reader=tf.TFRecordReader()

_,serialized_example=reader.read(filename_queue)

features = tf.parse_single_example(serialized_example, features = {
                                                               'label':tf.FixedLenFeature([], tf.int64),
                                                               'image_raw':tf.FixedLenFeature([], tf.string),
                                                           })

imagex = tf.image.decode_image(features['image_raw'], channels=3)
imagex.set_shape((182, 182, 3))
labelx=tf.cast(features['label'],tf.int32)

image_batch,label_batch=tf.train.batch(
    [imagex,labelx],batch_size=64,capacity=1000+3*64,num_threads=3

)


with tf.Session() as sess:

    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(sess=sess,coord=coord)

    for i in range(5):
        cur_image_batch,cur_label_batch=sess.run([image_batch,label_batch])

        print(cur_image_batch)

    coord.request_stop()
    coord.join(threads)















