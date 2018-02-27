from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19
from keras.layers.core import Flatten
from keras.models import Model
from keras.layers import Dense,Dropout
from keras.preprocessing import image


import numpy as np
import os
import json
import re
import matplotlib.pyplot as plt
import numpy as np
import warnings

from keras.layers import Input,merge
from keras.layers import Dense,Activation
from keras.layers import PReLU
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D,AveragePooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras.activations import softmax
from keras.optimizers import sgd
from keras.optimizers import adam
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.models import load_model
from keras.utils import plot_model
from sklearn import preprocessing

import keras
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.vgg16 import preprocess_input
#from keras.applications.inception_resnet_v2 import preprocess_input
from keras.layers import Dense, AveragePooling2D
from keras.utils import multi_gpu_model
from keras.layers import BatchNormalization,Activation
from keras.models import Model
import numpy



targetdir = os.listdir('/home/dany/Documents/lfw_aliged_224_224')

def get_mymodel():

    inp = Input(shape=(112, 96, 3))

    # Block 1
    #x = Conv2D(32, (3, 3), padding='same', name='block1_conv1',kernel_initializer="he_normal",kernel_regularizer=l2(5e-4))(inp)
    x = Conv2D(32, (3, 3), padding='same', name='block1_conv1', kernel_initializer="glorot_uniform",kernel_regularizer=l2(5e-4))(inp)
    x = BatchNormalization()(x)
    x =PReLU()(x)

    #x = Conv2D(64, (3, 3),  padding='same', name='block1_conv2',kernel_initializer="he_normal",kernel_regularizer=l2(5e-4))(x)
    x = Conv2D(64, (3, 3), padding='same', name='block1_conv2', kernel_initializer="glorot_uniform",kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)

    out1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    #x = Conv2D(64, (3, 3), padding='same', name='block2_conv1',kernel_initializer="he_normal",kernel_regularizer=l2(5e-4))(out1)
    x = Conv2D(64, (3, 3), padding='same', name='block2_conv1', kernel_initializer="glorot_uniform",kernel_regularizer=l2(5e-4))(out1)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    #x = Conv2D(64, (3, 3), padding='same', name='block2_conv2',kernel_initializer="he_normal",kernel_regularizer=l2(5e-4))(x)
    x = Conv2D(64, (3, 3), padding='same', name='block2_conv2', kernel_initializer="glorot_uniform",kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization()(x)
    x = merge([x, out1], mode='sum')
    x = PReLU()(x)
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #x = Conv2D(128, (3, 3), padding='same', name='block2_conv3',kernel_initializer="he_normal",kernel_regularizer=l2(5e-4))(x)
    x = Conv2D(128, (3, 3), padding='same', name='block2_conv3', kernel_initializer="glorot_uniform",kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)

    out2 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    #x = Conv2D(128, (3, 3), padding='same', name='block3_conv1',kernel_initializer="he_normal",kernel_regularizer=l2(5e-4))(out2)
    x = Conv2D(128, (3, 3), padding='same', name='block3_conv1', kernel_initializer="glorot_uniform",kernel_regularizer=l2(5e-4))(out2)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    #x = Conv2D(128, (3, 3), padding='same', name='block3_conv2',kernel_initializer="he_normal",kernel_regularizer=l2(5e-4))(x)
    x = Conv2D(128, (3, 3), padding='same', name='block3_conv2', kernel_initializer="glorot_uniform",kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization()(x)
    x = merge([x, out2], mode='sum')
    out3 = PReLU()(x)
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #x = Conv2D(128, (3, 3), padding='same', name='block3_conv3',kernel_initializer="he_normal",kernel_regularizer=l2(5e-4))(out3)
    x = Conv2D(128, (3, 3), padding='same', name='block3_conv3', kernel_initializer="glorot_uniform",kernel_regularizer=l2(5e-4))(out3)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    #x = Conv2D(128, (3, 3), padding='same', name='block3_conv4',kernel_initializer="he_normal",kernel_regularizer=l2(5e-4))(x)
    x = Conv2D(128, (3, 3), padding='same', name='block3_conv4', kernel_initializer="glorot_uniform",kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization()(x)
    x = merge([x, out3], mode='sum')
    x = PReLU()(x)
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #x = Conv2D(256, (3, 3), padding='same', name='block3_conv5',kernel_initializer="he_normal",kernel_regularizer=l2(5e-4))(x)
    x = Conv2D(256, (3, 3), padding='same', name='block3_conv5', kernel_initializer="glorot_uniform",kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)

    out4 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    #x = Conv2D(256, (3, 3), padding='same', name='block4_conv1',kernel_initializer="he_normal",kernel_regularizer=l2(5e-4))(out4)
    x = Conv2D(256, (3, 3), padding='same', name='block4_conv1', kernel_initializer="glorot_uniform",kernel_regularizer=l2(5e-4))(out4)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    #x = Conv2D(256, (3, 3), padding='same', name='block4_conv2',kernel_initializer="he_normal",kernel_regularizer=l2(5e-4))(x)
    x = Conv2D(256, (3, 3), padding='same', name='block4_conv2', kernel_initializer="glorot_uniform",kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization()(x)
    x = merge([x, out4], mode='sum')
    out5 = PReLU()(x)
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #x = Conv2D(256, (3, 3), padding='same', name='block4_conv3',kernel_initializer="he_normal",kernel_regularizer=l2(5e-4))(out5)
    x = Conv2D(256, (3, 3), padding='same', name='block4_conv3', kernel_initializer="glorot_uniform",kernel_regularizer=l2(5e-4))(out5)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    #x = Conv2D(256, (3, 3), padding='same', name='block4_conv4',kernel_initializer="he_normal",kernel_regularizer=l2(5e-4))(x)
    x = Conv2D(256, (3, 3), padding='same', name='block4_conv4', kernel_initializer="glorot_uniform",kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization()(x)
    x = merge([x, out5], mode='sum')
    out6 = PReLU()(x)
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #x = Conv2D(256, (3, 3), padding='same', name='block4_conv5',kernel_initializer="he_normal",kernel_regularizer=l2(5e-4))(out6)
    x = Conv2D(256, (3, 3), padding='same', name='block4_conv5', kernel_initializer="glorot_uniform",kernel_regularizer=l2(5e-4))(out6)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    #x = Conv2D(256, (3, 3), padding='same', name='block4_conv6',kernel_initializer="he_normal",kernel_regularizer=l2(5e-4))(x)
    x = Conv2D(256, (3, 3), padding='same', name='block4_conv6', kernel_initializer="glorot_uniform",kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization()(x)
    x = merge([x, out6], mode='sum')
    out7 = PReLU()(x)
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #x = Conv2D(256, (3, 3), padding='same', name='block4_conv7',kernel_initializer="he_normal",kernel_regularizer=l2(5e-4))(out7)
    x = Conv2D(256, (3, 3), padding='same', name='block4_conv7', kernel_initializer="glorot_uniform",kernel_regularizer=l2(5e-4))(out7)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    #x = Conv2D(256, (3, 3), padding='same', name='block4_conv8',kernel_initializer="he_normal",kernel_regularizer=l2(5e-4))(x)
    x = Conv2D(256, (3, 3), padding='same', name='block4_conv8', kernel_initializer="glorot_uniform",kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization()(x)
    x = merge([x, out7], mode='sum')
    out8 = PReLU()(x)
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #x = Conv2D(256, (3, 3), padding='same', name='block4_conv9',kernel_initializer="he_normal",kernel_regularizer=l2(5e-4))(out8)
    x = Conv2D(256, (3, 3), padding='same', name='block4_conv9', kernel_initializer="glorot_uniform",kernel_regularizer=l2(5e-4))(out8)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    #x = Conv2D(256, (3, 3), padding='same', name='block4_conv10',kernel_initializer="he_normal",kernel_regularizer=l2(5e-4))(x)
    x = Conv2D(256, (3, 3), padding='same', name='block4_conv10', kernel_initializer="glorot_uniform",kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization()(x)
    x = merge([x, out8], mode='sum')
    x = PReLU()(x)
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #x = Conv2D(512, (3, 3), padding='same', name='block4_conv11',kernel_initializer="he_normal",kernel_regularizer=l2(5e-4))(x)
    x = Conv2D(512, (3, 3), padding='same', name='block4_conv11', kernel_initializer="glorot_uniform",kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)

    out9 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    #Block 5
    #x = Conv2D(512, (3, 3), padding='same', name='block5_conv1',kernel_initializer="he_normal",kernel_regularizer=l2(5e-4))(out9)
    x = Conv2D(512, (3, 3), padding='same', name='block5_conv1', kernel_initializer="glorot_uniform",kernel_regularizer=l2(5e-4))(out9)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    #x = Conv2D(512, (3, 3), padding='same', name='block5_conv2',kernel_initializer="he_normal",kernel_regularizer=l2(5e-4))(x)
    x = Conv2D(512, (3, 3), padding='same', name='block5_conv2', kernel_initializer="glorot_uniform",kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization()(x)
    x = merge([x, out9], mode='sum')
    out10 = PReLU()(x)
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #x = Conv2D(512, (3, 3), padding='same', name='block5_conv3',kernel_initializer="he_normal",kernel_regularizer=l2(5e-4))(out10)
    x = Conv2D(512, (3, 3), padding='same', name='block5_conv3', kernel_initializer="glorot_uniform",kernel_regularizer=l2(5e-4))(out10)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    #x = Conv2D(512, (3, 3), padding='same', name='block5_conv4',kernel_initializer="he_normal",kernel_regularizer=l2(5e-4))(x)
    x = Conv2D(512, (3, 3), padding='same', name='block5_conv4', kernel_initializer="glorot_uniform",kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization()(x)
    x = merge([x, out10], mode='sum')
    out11 = PReLU()(x)
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #x = Conv2D(512, (3, 3), padding='same', name='block5_conv5',kernel_initializer="he_normal",kernel_regularizer=l2(5e-4))(out11)
    x = Conv2D(512, (3, 3), padding='same', name='block5_conv5', kernel_initializer="glorot_uniform",kernel_regularizer=l2(5e-4))(out11)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    #x = Conv2D(512, (3, 3), padding='same', name='block5_conv6',kernel_initializer="he_normal",kernel_regularizer=l2(5e-4))(x)
    x = Conv2D(512, (3, 3), padding='same', name='block5_conv6', kernel_initializer="glorot_uniform",kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization()(x)
    x = merge([x, out11], mode='sum')
    x = PReLU()(x)
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    x = Flatten(name='flatten')(x)
    x = Dense(512, name='fc1')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)

    final_out=Dense(10575, activation='softmax', name='fc2')(x)

    model = Model(inp, final_out)

    return model


model=get_mymodel()

#model.load_weights('/home/deep-visage/Desktop/epoch65_batch120sgd_glorot_My_deepvis.h5')

'''

base_model = InceptionResNetV2(weights=None, include_top=False, pooling='avg',input_shape=(299, 299, 3))

x=base_model.output

x = Dense(512, name='predictions')(x)

x = BatchNormalization()(x)

x = Activation('relu')(x)

logits = Dense(9131, name='logits',activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=logits)

model.load_weights('/home/deep-visage/checkpoints/22-26/InceptionResNetV2_vggface2_26.h5')
'''




#extrat_model=Model(inputs=model.input,outputs=model.get_layer('fc1').output)

testmodel=InceptionResNetV2(include_top=True, weights='imagenet',  input_shape=(224, 224, 3), pooling='avg')
#testmodel.load_weights('epoch_10.h5')

model=Model(inputs=testmodel.input,outputs=testmodel.get_layer('avg_pool').output)




features=[]

features_names=[]



count=0
for allDir in targetdir:
    child = os.path.join('%s%s' % ('/home/dany/Documents/lfw_aliged_224_224/', allDir))
    #child = os.path.join('%s%s' % ('/home/deep-visage/Desktop/lfw_299/', allDir))
    childx = os.listdir(child)
    for imagex in childx:

        count+=1
        print(count)
        img_path=os.path.join('%s%s%s' % ('/home/dany/Documents/lfw_aliged_224_224/', allDir + '/', imagex))
        #img_path = os.path.join('%s%s%s' % ('/home/deep-visage/Desktop/lfw_299/', allDir + '/', imagex))
        #print(img_path)
        img = image.load_img(img_path, target_size=(224,224))
        #img = image.load_img(img_path, target_size=(299, 299))
        # img=image.img_to_array(img)
        # img = img - 127.5
        # img = img / 128

        features.append(np.squeeze(model.predict(preprocess_input(np.expand_dims(image.img_to_array(img), axis=0))).tolist()))
        #features.append(np.squeeze(extrat_model.predict(preprocess_input(np.expand_dims(img, axis=0))).tolist()))
        features_names.append(imagex)

        #print(features[count-1])
        #print(features_names[count - 1])



features=preprocessing.normalize(features,norm='l2')

fea_dit=zip(features_names,features)
fea_dit=dict(fea_dit)

np.save("features.npy", np.array(features))
np.save("feature_names.npy", np.array(features_names))

positivescore = []
leftname = []
rightname = []
# positive
with open('/home/dany/Documents/pairs/positive', 'r') as f:
    data = f.readlines()  # txt中所有字符串读入data
    for line in data:
        name0 = re.split(r'[\s]', line)[0]
        name1 = re.split(r'[\s]', line)[1]
        name2 = re.split(r'[\s]', line)[2]
        # print(name0,' ',name1,' ',name2)
        if len(name1) == 1:
            file_left_name = name0 + '_000' + name1 + '.jpg'
        else:
            if len(name1) == 2:
                file_left_name = name0 + '_00' + name1 + '.jpg'
            else:
                file_left_name = name0 + '_0' + name1 + '.jpg'
        # print(file_left_name)

        if len(name2) == 1:
            file_right_name = name0 + '_000' + name2 + '.jpg'
        else:
            if len(name2) == 2:
                file_right_name = name0 + '_00' + name2 + '.jpg'
            else:
                file_right_name = name0 + '_0' + name2 + '.jpg'
        # print(file_right_name)
        # print(d[file_left_name])

        # print(d[file_right_name])
        leftname.append(file_left_name)
        rightname.append(file_right_name)

        #print(file_left_name,file_right_name)


        a = np.squeeze(np.array(fea_dit[file_left_name]))
        b = np.squeeze(np.array(fea_dit[file_right_name]))


        dot_product = np.dot(a, b)
        normA = np.dot(a, a)
        normB = np.dot(b, b)
        print(dot_product / ((normA * normB) ** 0.5))
        positivescore.append(dot_product / ((normA * normB) ** 0.5))






print('***************')
np.save("positivescore.npy", positivescore)
positivescore.sort()
#print(positivescore)
print(len(positivescore))



negativescore = []
leftname = []
rightname = []
#negative
with open('/home/dany/Documents/pairs/negative', 'r') as f:
    data = f.readlines()  # txt中所有字符串读入data
    for line in data:
        name0 = re.split(r'[\s]', line)[0]
        name1 = re.split(r'[\s]', line)[1]
        name2 = re.split(r'[\s]', line)[2]
        name3 = re.split(r'[\s]', line)[3]
        #print(name0,' ',name1,' ',name2,' ',name3)

        if len(name1) == 1:
            file_left_name = name0 + '_000' + name1+'.jpg'
        else:
            if len(name1) == 2:
                file_left_name = name0 + '_00' + name1+'.jpg'
            else:
                file_left_name = name0 + '_0' + name1+'.jpg'
                # print(file_left_name)


        if len(name3)==1 :
            file_right_name=name2+'_000'+name3+'.jpg'
        else:
            if len(name3) == 2:
                file_right_name = name2 + '_00' + name3+'.jpg'
            else:
                file_right_name = name2 + '_0' + name3+'.jpg'

        leftname.append(file_left_name)
        rightname.append(file_right_name)

        #a = np.array(fea_dit[file_left_name]).flatten()
        #b = np.array(fea_dit[file_right_name]).flatten()
        a = np.squeeze(np.array(fea_dit[file_left_name]))
        b = np.squeeze(np.array(fea_dit[file_right_name]))




        dot_product = np.dot(a, b)
        normA = np.dot(a, a)
        normB = np.dot(b, b)
        print(dot_product / ((normA * normB) ** 0.5))
        negativescore.append(dot_product / ((normA * normB) ** 0.5))




print('***************')
np.save("negativescore.npy", negativescore)
negativescore.sort()
print(len(negativescore))



'''
jsObj = json.dumps(fea_dit)
fileObject = open('/home/dany/Desktop/jsonFile.json', 'a+')
fileObject.write(jsObj)
fileObject.close()
'''

positivescore=np.load("positivescore.npy")
negativescore=np.load("negativescore.npy")
len_ps=len(positivescore)
len_ns=len(negativescore)


minscore=min(positivescore.min(),negativescore.min())
maxscore=max(positivescore.max(),negativescore.max())
minimun=round(minscore,1)
maxmun=round(maxscore,1)


Tar=[]
Far=[]


for threshold in np.linspace(0,1,10001):
    if threshold == 0  or threshold == 1 :
        pass
    else:
        #print(threshold)


        tp=np.sum(positivescore>threshold)
        fn=len_ps-tp
        fp=np.sum(negativescore>threshold)
        tn=len_ns-fp
        tar=tp/(tp+fn)
        far=fp/(fp+tn)
        acc=(tp+tn)/(len_ns+len_ps)




        print('%%%%%%%%%%%%%%%%%%%%%%%')
        print(threshold,tp,fn,fp,tn,tar,far,acc)
        Tar.append(tar)
        Far.append(far)

Tar.reverse()
Far.reverse()

plt.plot(Far,Tar,linewidth=1.5)
plt.show()






