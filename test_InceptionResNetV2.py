from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from keras.layers import Dense, AveragePooling2D,Convolution2D, MaxPooling2D,Activation,GlobalAveragePooling2D
from sklearn.metrics import log_loss
from keras.models import Model
from keras.datasets import cifar10
from keras.layers.core import Flatten
import keras
from keras.preprocessing.image import ImageDataGenerator

from keras.applications.resnet50 import ResNet50
from keras.applications.xception import Xception
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2

'''
base_model = InceptionV3(weights=None,include_top=True,input_shape=(224,224,3),classes=10575)
base_model.load_weights('/home/dany/checkpoints/4-14/weights-improvement-09-0.7433-1.7867.hdf5')
'''
base_model = InceptionResNetV2(weights=None,include_top=True,input_shape=(224,224,3),classes=10575)
#base_model.load_weights('/home/dany/checkpoints/4-14/weights-improvement-09-0.7433-1.7867.hdf5')



base_model.compile(optimizer=keras.optimizers.RMSprop(lr=0.01, rho=0.9, epsilon=1e-06), loss='categorical_crossentropy',metrics=['accuracy'])

data_dir='/home/dany/Desktop/CASIA-WebFace_grey_224'
test_data_dir='/home/dany/Desktop/CASIA-WebFace_grey_224_test'

#batch_size = 80
batch_size = 70


datagen = ImageDataGenerator(
        width_shift_range=0.08,
        height_shift_range=0.08,
        zoom_range=0.1,
        horizontal_flip=True,
        zca_whitening=True
)

generator = datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        shuffle=True
)

generator_test= datagen.flow_from_directory(
        test_data_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        shuffle=True
)


#early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',mode='min',patience=2,verbose=0)
#checkpointer = keras.callbacks.ModelCheckpoint(filepath='/home/dany/checkpoints/weights-improvement-{epoch:02d}-{val_acc:.4f}-{val_loss:.4f}.hdf5', monitor='val_loss',mode='min',verbose=0, save_best_only=True)
checkpointer = keras.callbacks.ModelCheckpoint(filepath='/home/dany/checkpoints/weights-improvement-{epoch:02d}-{val_acc:.4f}-{val_loss:.4f}.hdf5', monitor='val_loss',mode='min',verbose=0, save_best_only=False)
#reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1,verbose=0, mode='min', epsilon=0.0001,cooldown=0, min_lr=0.00001)
logger=keras.callbacks.CSVLogger('/home/dany/checkpoints/log.csv', separator=',', append=False)

base_model.fit_generator(generator,epochs=5,steps_per_epoch=13082,validation_data=generator_test,validation_steps=708,callbacks=[logger,checkpointer])

#base_model.save('keras_xception_webface_0.h5')



