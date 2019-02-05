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

After using the previously computed weights to predict the classes of these new images, the accuracy came out to a dismal 40%, compared to the 93% accuracy on the original test set. Granted, the new set of five images isn't a statistically significant number

### Visualizing the Layers of the CNN

### Future Work
