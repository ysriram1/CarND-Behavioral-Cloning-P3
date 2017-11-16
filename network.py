import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import random
from sklearn.utils import shuffle

# set keras backend to tensorflow
os.environ['KERAS_BACKEND'] = 'tensorflow'

# set flags
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('home_dir',
'C:/Users/Sriram/Desktop/Udacity/selfdriving_car/CarND-Behavioral-Cloning-P3',
"location of folder that contains the data folder")
flags.DEFINE_integer('batch_size', 128, "size of each batch")
flags.DEFINE_integer('epochs', 20, "number of epochs to run the training")
flags.DEFINE_float('learn_rate', 0.0001, "learning rate to use for training")

# set home directory
os.chdir(FLAGS.home_dir)

# extract required image loc data from list
# TODO: expand func to include other targers (throttle brake etc)
def extract_image_loc_data(all_data, center=True,
                            right=False, left=False,
                            train_valid_split=True,
                            train_size=0.7):
    image_loc_lst = []
    for line in all_data:
        c,l,r,angle,throt,brake,speed = line.split(',')
        angle = float(angle)
        if center: image_loc_lst.append([c,angle])
        if right: image_loc_lst.append([r,side_angle(angle,side='r')])
        if left: image_loc_lst.append([l,side_angle(angle,side='l')])

    random.shuffle(image_loc_lst) # shuffle in-place
    if train_valid_split:
        train_end_idx = int(len(image_loc_lst)*train_size)
        return image_loc_lst[0:train_end_idx], image_loc_lst[train_end_idx:-1]
    return image_loc_lst

# batch data generator
# img_loc_response_list: a list of the location of a given Image
#               and its corresponding class (target) value
# this generator get actual images from the ./data/IMG folder
def batch_data_gen(img_loc_response_list, batch_size):

    loc_list = shuffle(img_loc_response_list)
    data_size = len(loc_list)

    while True:
        for start_batch in range(0, data_size, batch_size):
            end_batch = start_batch + batch_size
            if end_batch > data_size: end_batch = data_size
            loc_list_batch = loc_list[start_batch:end_batch]
            X_batch_loc, y_batch = zip(*loc_list_batch)
            X_batch = list(map(lambda x: plt.imread(x),
                                X_batch_loc)) # extract images from specified location

            yield np.array(X_batch), np.array(y_batch)

# add correction to right or left camera angle
def side_angle(angle, side, correction=0.2):
    if side=='r': return angle - correction
    if side=='l': return angle + correction

# obtain data list with image location and target value

with open('./data3/driving_log.csv', 'r') as f:
    read_file = f.readlines()
    train_data_lst, valid_data_lst =\
     extract_image_loc_data(read_file, center=True,
                            left=False, right=False, # not using left and right cameras
                            train_valid_split=True)


train_gen = batch_data_gen(img_loc_response_list=train_data_lst,
                                batch_size=FLAGS.batch_size)
valid_gen = batch_data_gen(img_loc_response_list=valid_data_lst,
                                batch_size=FLAGS.batch_size)
# model using keras
from keras.models import Sequential
from keras.layers import Lambda, Dense, Flatten, Dropout, Activation
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam

model = Sequential()

row, col, ch = 160, 320, 3

# preprocessing layers
model.add(Lambda(lambda x: x/127.5 + 1,
                input_shape=(row,col,ch),
                output_shape=(row,col,ch))) # normalize the data
model.add(Cropping2D(cropping=((50,20),(0,0)),
                    input_shape=(row,col,ch))) # crop image

# TODO: edit below to create a more powerful model
# a model similar to nvidias (from paper)
# conv layers
# conv1
model.add(Conv2D(filters=24, kernel_size=(5,5),
                strides=(2,2), padding='same'))
model.add(Activation('relu'))

# conv2
model.add(Conv2D(filters=36, kernel_size=(5,5),
                strides=(2,2), padding='same'))
model.add(Activation('relu'))

# conv3
model.add(Conv2D(filters=48, kernel_size=(5,5),
                strides=(2,2), padding='same'))
model.add(Activation('relu'))

# conv4
model.add(Conv2D(filters=64, kernel_size=(3,3),
                strides=(1,1), padding='valid'))
model.add(Activation('relu'))

# conv5
model.add(Conv2D(filters=36, kernel_size=(5,5),
                strides=(2,2), padding='valid'))
model.add(Activation('relu'))

# Flatten
model.add(Flatten())

# fully connected 1
model.add(Dense(500)) # a linear regression network

# Dropout
model.add(Dropout(0.1))

# fully connected 2
model.add(Dense(150))

# fully connected 3
model.add(Dense(10))

# final output layer
model.add(Dense(1))

# train the model
optim = Adam(lr=FLAGS.learn_rate)
model.compile(loss='mse',optimizer=optim,metrics=['mse'])

model.fit_generator(generator=train_gen,
                    steps_per_epoch=len(train_data_lst)//FLAGS.batch_size,
                    epochs=FLAGS.epochs,
                    validation_data=valid_gen,
                    validation_steps=len(valid_data_lst)//FLAGS.batch_size)

model.save('model_2_gpu.h5')
