# **Behavioral Cloning Project** 


### In this project, I trained a convolutional neural network (CNN) to learn from simulated driving data and then drive itself around a test track.

---

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build a CNN in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around the track one without leaving the road

A visualization of the trained network driving itself around the test track is shown below.

(insert gif)

[//]: # (Image References)

[image0a]: ./imgs/center.jpg "center"
[image0b]: ./imgs/left.jpg "left"
[image0c]: ./imgs/right.jpg "right"
[image1]: ./imgs/log_file.png "drive log"
[image2]: ./imgs/pre-augment.jpg "pre augmentation"
[image3]: ./imgs/post-augment.jpg "post augmentation"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

### Project Files
My project includes the following files:
* `model.py` containing the script to create and train the model
* `drive.py` for driving the car in autonomous mode
* `model.h5` containing a trained convolution neural network 
* `README.md` summarizing the results 

### Simulation Data
To begin training my model, I used the provided Udacity dataset which consists of camera images of the car driving around the track. There are three cameras mounted left, center, and right of the windshield that simultaneously collect data. 

![alt text][image0a] ![alt text][image0b] ![alt text][image0c]

Having the three cameras helps the network generalize while training. Furthermore, having the left and right images help with correcting the steering angle when the car is off center, as the recorded steering angle corresponds to the center camera image. Therefore, when training with the left and right camera images, I added a correction factor that adjusted the steering angle depending on which camera image was used. The log file containing the driving data is shown below (only the steering angle data was used as an output label).

![alt text][image1]

### Model Architecture
My model is a modified version of the CNN proposed by the [comma.ai team](https://github.com/commaai/research/blob/master/train_steering_model.py). The model consists of three convolutional layers followed by a fully connected layer followed by a final output layer. I used batch normalization

The network takes as input a three channel (RGB) color image of height x pixels and width y pixels. It 

with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

### More Data / Data Augmentation
After training the model weights on the provided dataset, I tweaked the model further by collecting more simulation data. I drove backwards around the track in order to combat the left-turn bias present in track one. I also drove off-center and weaved left and right in order to train the network how to respond when it goes off to the side of the road.

As seen in the figure below, most of the steering angle data is zero because there are large portions of the track that are straight. However, this could cause the model to overfit to those straight-line cases and struggle on turns.

![alt text][image2]

To combat this, I augmented the dataset by flipping the image horizontally and taking the negative of the steering measurement for any cases where the steering angle was not zero. The results of this augmentation are shown below.

![alt text][image3]
