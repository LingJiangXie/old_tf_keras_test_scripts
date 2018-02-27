from datetime import datetime
import  os
import tensorflow as tf
import time
import facenet
import inception_resnet_v1

batch_size=64
lr=0.01
lr_decay=0.99
regular_rate=0.0001
epoch=10
no_gpu=1
mad=0.99
image_height=160
image_width=160

keep_probability=0.8

phase_train=True

embedding_size=256

weight_decay=5e-5

model_save_dir='/home/dany/checkpoints/'

model_name='model.ckpt'

data_path='/home/dany/Documents/workspace/tensor_utils/webface_182.tfrecords'

#input_queue

def get_input():
    filename_queue = tf.train.string_input_producer([data_path])
    reader =tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example, features={
        'label': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string),
    })

    imagex = tf.image.decode_image(features['image_raw'], channels=3)
    imagex.set_shape((182, 182, 3))


    imagex = tf.py_func(facenet.random_rotate_image, [imagex], tf.float32)
    imagex = tf.random_crop(imagex, [image_height, image_width, 3])
    imagex = tf.image.resize_image_with_crop_or_pad(imagex, image_height, image_width)
    image = tf.image.random_flip_left_right(imagex)

    label = tf.cast(features['label'], tf.int32)

    min_after_dequeue=10000
    capacity =min_after_dequeue+3*batch_size
    return tf.train.shuffle_batch([image,label], batch_size=batch_size,
            capacity=capacity,min_after_dequeue=min_after_dequeue,
            allow_smaller_final_batch=True)

def get_loss(x,labels,scope):

    y=inception_resnet_v1.inference(x, keep_probability=keep_probability,
            phase_train=phase_train, bottleneck_layer_size=embedding_size,
            weight_decay=weight_decay)

    cross_entropy=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=y))


    regularization_loss=tf.add_n(tf.get_collection('losses',scope))

    loss=cross_entropy+regularization_loss

    return loss

def average_gd(tower_grads):
    average_gds=[]
    for grad_and_vars in zip(*tower_grads):

        grads=[]
        for g,_ in grad_and_vars:
            expanded_g=tf.expand_dims(g,0)
            grads.append(expanded_g)

        grad=tf.concat(0,grads)
        grad=tf.reduce_mean(grad,0)

        v=grad_and_vars[0][1]

        grad_and_var=(grad,v)
        average_gds.append(grad_and_var)

    return average_gds

def main(argv=None):
    with tf.Graph().as_default(),tf.device('/cpu:0'):

        x,y_=get_input()
        global_step=tf.get_variable('global_step',[],initializer=tf.constant_initializer(0),trainable=False)

        learning_rate=tf.train.exponential_decay(lr,global_step,3341/batch_size,lr_decay)

        opt=tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)

        tower_grads=[]

        for i in range(no_gpu):
            with tf.device('/gpu:%d'%i) :
                with tf.name_scope('GPU_%d'%i) as scope:
                    cur_loss=get_loss(x,y_,scope)
                    tf.get_variable_scope().reuse_variables()

                    grads=opt.compute_gradients(cur_loss)
                    tower_grads.append(grads)

        grads=average_gd(tower_grads)

        apply_gradient_op=opt.apply_gradients(grads,global_step=global_step)

        variable_avg=tf.train.ExponentialMovingAverage(mad,global_step)

        variables_avg_op=variable_avg.apply(tf.trainable_variables())

        train_op=tf.group(apply_gradient_op,variables_avg_op)

        saver=tf.train.Saver(tf.all_variables())

        init=tf.initialize_all_variables()

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)) as sess:
            init.run()
            coord=tf.train.Coordinator()
            therads=tf.train.start_queue_runners(sess=sess,coord=coord)

            for step in range(epoch):
                start_time=time.time()
                _,loss_value=sess.run([train_op,cur_loss])
                duration=time.time()-start_time

                num_examples_per_step=batch_size*no_gpu

                examples_per_sec=num_examples_per_step/duration

                sec_per_batch=duration/no_gpu

                format_str=('step%d,loss=%.2f(%.1f examples/'' sec;%.3f sec/batch)')

                print(format_str % (step,loss_value,examples_per_sec,sec_per_batch))

                if step%2==0 or (step+1)==epoch:
                    checkpoint_path=os.path.join(model_save_dir,model_name)
                    saver.save(sess,checkpoint_path,global_step=global_step)

            coord.request_stop()
            coord.join(therads)

if __name__ =='__main__':
    tf.app.run()





