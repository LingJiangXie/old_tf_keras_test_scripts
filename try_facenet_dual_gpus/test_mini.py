import facenet
import tensorflow as tf
import os
import inception_resnet_v1
import tensorflow.contrib.slim as slim

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

main_path='/home/deep-visage/Documents/mini_images'

data_path='/home/deep-visage/Documents/mini_images_224.tfrecords'

batch_size=64

'''

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

writer=tf.python_io.TFRecordWriter(data_path)

class_name_list=os.listdir(main_path)
count=0
for index,classname in enumerate(class_name_list):
    class_path=os.path.join(main_path,classname)
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
'''

def get_input():
    filename_queue = tf.train.string_input_producer([data_path])
    reader =tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example, features={
        'label': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string),
    })

    imagex = tf.image.decode_image(features['image_raw'], channels=3)
    imagex.set_shape((224, 224, 3))



    imagex = tf.py_func(facenet.random_rotate_image, [imagex], tf.uint8)
    imagex = tf.random_crop(imagex, [160, 160, 3])
    imagex = tf.image.resize_image_with_crop_or_pad(imagex, 160, 160)
    image = tf.image.random_flip_left_right(imagex)

    image = tf.cast(image, tf.float32)

    label = tf.cast(features['label'], tf.int32)

    min_after_dequeue=10000
    capacity =min_after_dequeue+3*batch_size
    return tf.train.shuffle_batch([image,label], batch_size=batch_size,
            capacity=capacity,min_after_dequeue=min_after_dequeue,
            allow_smaller_final_batch=True)


with tf.Graph().as_default(),tf.device('/cpu:0'):

    init = tf.global_variables_initializer()

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

    sess.run(init)
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(coord=coord, sess=sess)

    x,y=get_input()

    prelogits, _ = inception_resnet_v1.inference(x, keep_probability=0.8,
                                                 phase_train=True, bottleneck_layer_size=512,
                                                 weight_decay=5e-5)
    logits = slim.fully_connected(prelogits, 8, activation_fn=None,
                                  weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                  weights_regularizer=slim.l2_regularizer(5e-5),
                                  scope='Logits', reuse=False)

    embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(coord=coord, sess=sess)

    with sess.as_default():





        # print(x)



        '''
        print(prelogits) #Tensor("InceptionResnetV1/Bottleneck/BatchNorm/batchnorm/add_1:0", shape=(?, 512), dtype=float32)
        print(x) #Tensor("shuffle_batch:0", shape=(?, 160, 160, 3), dtype=float32)
        '''
        print(embeddings.eval(session=sess))




