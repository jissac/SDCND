"""
Steering angle prediction model based on the comma.ai implementation
"""
import argparse
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Conv2D
from keras import backend as K 
from keras.utils import plot_model
import numpy as numpy
from pandas import read_csv
import matplotlib.pyplot as plt


def cnn_model():
    ch, row, col = 3,320,160
    
    model = Sequential()
    model.add(Lambda(lambda x: x/255.,
                     input_shape=(row,col,ch),
                     output_shape=(row,col,ch)))
    model.add(Conv2D(filters=16,kernel_size=(8,8),strides=(4,4),padding='SAME'))
    model.add(ELU())
    model.add(Conv2D(filters=32,kernel_size=(5,5),strides=(2,2),padding='SAME'))
    model.add(ELU())
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


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Steering angle model trainer')
  parser.add_argument('--host', type=str, default="localhost", help='Data server ip address.')
  parser.add_argument('--port', type=int, default=5557, help='Port of server.')
  parser.add_argument('--val_port', type=int, default=5556, help='Port of server for validation dataset.')
  parser.add_argument('--batch', type=int, default=64, help='Batch size.')
  parser.add_argument('--epoch', type=int, default=200, help='Number of epochs.')
  parser.add_argument('--epochsize', type=int, default=10000, help='How many frames per epoch.')
  parser.add_argument('--skipvalidate', dest='skipvalidate', action='store_true', help='Multiple path output.')
  parser.set_defaults(skipvalidate=False)
  parser.set_defaults(loadweights=False)
  args = parser.parse_args()

  model = get_model()
  model.fit_generator(
    gen(20, args.host, port=args.port),
    samples_per_epoch=10000,
    nb_epoch=args.epoch,
    validation_data=gen(20, args.host, port=args.val_port),
    nb_val_samples=1000
  )
  print("Saving model weights and configuration file.")

  if not os.path.exists("./outputs/steering_model"):
      os.makedirs("./outputs/steering_model")

  model.save_weights("./outputs/steering_model/steering_angle.keras", True)
  with open('./outputs/steering_model/steering_angle.json', 'w') as outfile:
    json.dump(model.to_json(), outfile)