## Traffic Sign Classification Project

### In this project, I used convolutional neural nets to classify traffic sign images from the German Traffic Sign Database.

---

The code for this project is contained in the file `Traffic_Sign_Classifier_v1.ipynb`

**Outline**
* Dataset Exploration - summarizing and visualizing the data
* Preprocessing the training and validation sets
* Building the model architecture
* Training the model and obtaining validation set accuracy
* Using training weights to obtain accuracy on test set
* Testing the model on new traffic sign images obtained from the internet
* Visualizing the Convolutional Neural Net (CNN) hidden layers 

![](adv_lane_line.gif)

[//]: # (Image References)

[image1]: ./output/examples.jpg "examples"
[image2]: ./output/training_hist.jpg "training_hist"
[image3]: ./output/valid_hist.jpg "valid_hist"
[image4]: ./output/test_hist.jpg "test_hist"
[image5]: ./output/normalized.jpg "normalized"
[image6]: ./output/web_images.jpg "web_images"
[image7]: ./output/Feature_viz.png "feature_viz"

### Dataset Exploration
The [German traffic dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) is a set of 50,000 images, each image containing one traffic sign each, with a total of 43 classes. Below are some example images from the dataset.

![alt text][image1]

There are a total of 34799 training examples, 4410 validation examples, and 12630 test examples. Each image is 32 pixels by 32 pixels with 3 color channels (RGB).  The graphs below show the distribution of images across each of the 43 classes for the training, validation, and test sets.

![alt text][image2] ![alt text][image3] ![alt text][image4]

### Design and Test Model Architecture
I started by converting the images to grayscale and preprocessing the training and validation sets in order to make it easier for the CNN to learn during training: by normalizing the images to have a mean close to zero and equal variance, the feature values are distributed evenly which makes it easier for the optimizer to find a good solution. The sample of the normalized images is shown below:

![alt text][image5]

I initialized the weights using `tf.truncated_normal`, which generates normally distributed values with a mean of `mu` and standard deviation of `sigma`. It also drops any values that are more than 2 standard deviations from the mean and re-calculates them. This is important for training as extreme weight values would cause the gradients to unevenly update during backpropagation. 

The model I used was based on the [LeNet architecture](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf) proposed by Yann LeCun. This CNN architecture is five-layer network with two convolutional layers and three fully connected layers. For my architecture, I modified the LeNet and used three convolutional layers and two fully connected layers and added regularized the network by adding dropout before the last two layers (the convolutional and pooling layers are considered as one layer):

Input image --> conv1 --> pooling1 --> conv2 --> pooling2 --> conv3 --> pooling 3 --> fc1 --> fc2 --> logits

After 15 ephocs of training, the validation accuracy was 96%. The test set accuracy was 93.2%.

### Testing on New Images
After training on the given set of images, I found five new images of German traffic signs from the web. The normalized images are shown below:

![alt text][image6]

After using the previously computed weights to predict the classes of these new images, the accuracy came out to only 40% (2 images correctly classified), compared to the 93% accuracy on the original test set. Increasing the number of examples in this new test set and making sure the quality of the images is consistent with the training set (traffic signs centered in image, not at an angle or too pixelated) would help with the new image classification accuracy. 

The top five softmax probabilities for each of the images are shown below. 

`TopKV2(values=array([[  5.68378150e-01,   2.29687467e-01,   1.72620669e-01,
          2.05079950e-02,   4.13777446e-03],
       [  9.99371350e-01,   2.41140224e-04,   1.92313732e-04,
          1.89768194e-04,   1.65519373e-06],
       [  9.54150975e-01,   3.95272262e-02,   4.81002731e-03,
          1.34040369e-03,   1.70177722e-04],
       [  1.00000000e+00,   3.07106380e-16,   3.06713711e-19,
          2.72498174e-20,   2.09481934e-23],
       [  9.99996305e-01,   2.35562629e-06,   1.02180149e-06,
          1.49431941e-07,   8.77318058e-08]], dtype=float32), indices=array([[33, 15, 13, 35, 18],
       [19, 40, 37, 11, 10],
       [40, 12,  4,  1, 26],
       [17,  9, 14, 10, 34],
       [25, 24, 22, 31, 29]], dtype=int32))
GroundTruth:  [ 4 23 27 17 25]`

Comparing the top predicted labels [33 19 40 17 25] with the ground truth values of [4 23 27 17 25], we confirm that the model got the last two images correct. Since the softmax probabilities all add to one, we can see that the model was very sure of it's predictions for images 2,3,4,and 5 (.99 or greater), allbeit incorrectly. 

### Visualizing the Layers of the CNN
After training the CNN it's useful to show how the feature maps look like by plotting the output of the network's weight layers in response to a test stimuli image. From these plotted feature maps, it's possible to see what characteristics of an image the network found interesting. It's interesting to see the higher-level features and lower-level edges and lines that the CNN found important, as shown from the conv1 and conv2 visualizations below.

![alt text][image7]
### Future Work
To improve accuracy, I can use a deeper network that includes more convolutional layers and train on more images. From the histogram, there are certain classes of images that are under-represented in the training set - the CNN can be better trained by adding more images from these under-represented classes to the training set (through image augmentation techniques). 
