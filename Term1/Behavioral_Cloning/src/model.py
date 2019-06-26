"""
Steering angle prediction model based on the comma.ai implementation
"""
import csv
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers import Conv2D
from keras import backend as K 
from keras.utils import plot_model
from keras.layers import BatchNormalization
from keras.layers import Cropping2D
from keras import optimizers
from keras.models import load_model
import cv2

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt


def cnn_model():
    row, col, ch = 320,160,3
    
    model = Sequential()
    model.add(Cropping2D(cropping=((60,20), (0,0)), input_shape=(col,row,ch)))

    model.add(Lambda(lambda x: x/255. - 0.5))
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
    
    adam = optimizers.Adam(lr=1e-5)
    model.compile(optimizer=adam,loss='mse')
    
    return model

def load_csv_log(filepath_log):
    '''
    Loads driving log into a dataframe
    '''
    lines = []
    with open(filepath_log) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
            
    return lines

def split_data(log_df,split_ratio=0.2):
    '''
    Shuffles and splits log data into training and validation sets
    '''
    train, validation = train_test_split(log_df,test_size=split_ratio)

    return train, validation

def process_imgs(batch_sample):
    '''
    Loads and processes images from csv file, assigns steering angles to each image
    '''
    steering_angle = np.float32(batch_sample[3])
    # print(steering_angle)
    images, steering_angles = [],[]
    correction_factor = 0.15
    
    for camera_location in range(3):
        name = './data/track_full/IMG/' + batch_sample[camera_location].split('/')[-1]
        # print(name)
        image = cv2.imread(name)
        image_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        
        images.append(image_rgb)
        
        if camera_location == 1: # left image
            steering_angles.append(steering_angle + correction_factor)
        elif camera_location == 2: # right image
            steering_angles.append(steering_angle - correction_factor)
        else:
            steering_angles.append(steering_angle)
            if steering_angle != 0:
                flipped_center_image = cv2.flip(image_rgb, 1)
                images.append(flipped_center_image)
                steering_angles.append(-steering_angle)

    return images, steering_angles

def augment_imgs():
    '''
    Augment dataset
    '''
#     if camera_location == 0 & steering_angle != 0:
#         flipped = np.fliplr(image_rgb)
#         images.append(flipped_image)
#         steering_angles.append(-steering_angle)

    return None

def generator(samples, batch_size):
    '''
    Generates data in batches for the model
    '''
    num_samples = len(samples)
    while True:
        shuffle(samples)
        for offset in range(0,num_samples,batch_size):
            batch_samples = samples[offset:offset+batch_size]
            #print(batch_samples)
            #print('.....................samples........................')
            for batch_sample in batch_samples:
                # print(batch_sample)
                # print('////////////////sample///////////////////')
                images, steering_angles = process_imgs(batch_sample)
            X_train, y_train = np.array(images), np.array(steering_angles)
            yield shuffle(X_train, y_train)

if __name__ == "__main__":
    '''
    Train and save the model
    '''
    EPOCHS = 7
    BS = 128
    model = cnn_model()
#     model = load_model('model_track1_full.h5')
    model.load_weights('model_weights_track_full2-06-01.h5')
    #model.summary()
    log_file = load_csv_log('./data/track_full/driving_log.csv')
    # print(log_file[1])
    # print(log_file[1][1])
    train_samples, validation_samples = split_data(log_file[1:])
    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=BS)
    # print('traingen')
    validation_generator = generator(validation_samples, batch_size=BS)
    # print('valgen')
    model_history = model.fit_generator(generator=train_generator,
                        steps_per_epoch=len(train_samples)//BS,
                        validation_data=validation_generator,
                        validation_steps=len(validation_samples),
                        epochs=EPOCHS, verbose=1)
    model.save('model-06-13.h5')
    model.save_weights('model_weights-06-13.h5')
    #plot_model(model, to_file='model_plot.png',show_shapes=True, show_layer_names=True)

#     ### print the keys contained in the history object
#     print(model_history.history.keys())

#     ### plot the training and validation loss for each epoch
#     plt.plot(model_history.history['loss'])
#     plt.plot(model_history.history['val_loss'])
#     plt.title('model mean squared error loss')
#     plt.ylabel('mean squared error loss')
#     plt.xlabel('epoch')
#     plt.legend(['training set', 'validation set'], loc='upper right')
#     plt.savefig('loss_track1_recovers.jpg')
