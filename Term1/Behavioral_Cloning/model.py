"""
Steering angle prediction model based on the comma.ai implementation
"""
import argparse
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Conv2D
from keras import backend as K 
from keras.utils import plot_model
from keras.layers import BatchNormalization
from keras.layers import Cropping2D
import cv2

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as numpy
from pandas import read_csv
import matplotlib.pyplot as plt


def cnn_model():
    row, col, ch = 320,160,3
    
    model = Sequential()
    model.add(Cropping2D(cropping=((60,20), (0,0)), input_shape=(row,col,ch)))

    model.add(Lambda(lambda x: x/255.,
                     input_shape=(row,col,ch)))
    model.add(Conv2D(filters=16,kernel_size=(8,8),strides=(4,4),padding='SAME'))
    model.add(ELU())
    
    model.add(BatchNormalization())
    model.add(Conv2D(filters=32,kernel_size=(5,5),strides=(2,2),padding='SAME'))
    model.add(ELU())
    
    model.add(BatchNormalization())
    model.add(Conv2D(filters=64,kernel_size=(5,5),strides=(2,2),padding='SAME'))

    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(ELU())

    model.add(Dense(512))
    model.add(Dropout(0.5))
    model.add(ELU())

    model.add(Dense(1))
    
    model.compile(optimizer='adam',loss='mse')
    
    return model

def load_csv_log(filepath_log):
    '''
    Loads driving log into a dataframe
    '''
    log_df = read_csv(filepath_log)

    return log_df

def split_data(log_df,split_ratio=0.2):
    '''
    Shuffles and splits log data into training and validation sets
    '''
    train, validation = train_test_split(log_df,test_size=split_ratio)

    return train, validation

def augmentor(batch_sample):
    '''
    Crops, resizes, and horizontally flips each image
    '''
    steering_angle = np.float32(batch_sample[3])
    images, steering_angles = [],[]
    for camera_location in range(3):
        name = './IMG/'+batch_sample[camera_location].split('/')[-1]
        print(name)
        image = cv2.imread(name)
        image_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        
        images.append(image_rgb)
        
        if camera_location == 1:
            steering_angles.append(steering_angle + 0.2)
        elif camera_location == 2:
            steering_angles.append(steering_angle - 0.2)
        else:
            steering_angles.append(steering_angle)
        
        if camera_location == 0 & steering_angle != 0:
            flipped = np.fliplr(image_rgb)
            images.append(flipped_image)
            steering_angles.append(-steering_angle)
    return images, steering_angles

def generator(samples, batch_size=2):
    '''
    Function that generates data for the model
    '''
    num_samples = len(samples)
    while True:
        shuffle(samples)
        for offset in range(0,num_samples,batch_size):
            batch_samples = samples.iloc[offset:offset+batch_size]
            #print(batch_samples)
            for i, batch_sample in batch_samples.iterrows():
                images, steering_angles = augmentor(batch_sample)
            X_train, y_train = np.array(images), np.array(steering_angles)
            yield shuffle(X_train, y_train)

if __name__ == "__main__":
    '''
    Train and save the model
    '''
    model = cnn_model()
    log_file = load_csv_log('./driving_log.csv')
    train, validation = split_data(log_file[0:20])
    #print(train.head())
    model.fit_generator(generator=generator(train),steps_per_epoch=len(train),epochs=1, verbose = 1)

