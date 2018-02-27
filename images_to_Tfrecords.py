import tensorflow as tf
import os

main_path='/home/deep-visage/Documents/mini_images'

filename='/home/dany/DeskTop/CASIA-WebFace_grey_224.tfrecords'

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))



class_name_list=os.listdir(main_path)

count=0
#maybe not 600
per_tfrecord_nums=1024

#tf_record_id=count/per_tfrecord_nums
#writer = tf.python_io.TFRecordWriter('mini_images.tfrecords')


for index,classname in enumerate(class_name_list):
    class_path=os.path.join(main_path,classname)

    writer = tf.python_io.TFRecordWriter(
        'mini_test-%.5d.tfrecords' % (count / per_tfrecord_nums))
        

    for image_name in os.listdir(class_path):

        image_path=os.path.join(class_path,image_name)
        image_raw=tf.gfile.FastGFile(image_path,'rb').read()
        example=tf.train.Example(features=tf.train.Features(feature={
            'label' :_int64_feature(index),
            'image_raw':_bytes_feature(image_raw)
        }))
        writer.write(example.SerializeToString())
        count+=1
        print(count)

writer.close()