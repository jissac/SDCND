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

[image1]: ./img/output_images/cal_undist.jpg "cal_undist"
[image2]: ./img/output_images/test_undist.jpg "test_undist"

### Dataset Exploration
The [German traffic dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) is a set of 50,000 images, each image containing one traffic sign each, with a total of 43 classes. 
![alt text][image1]

### Design and Test Model Architecture
I started by preprocessing the training and validation sets in order to make it easier for the CNN to learn during training: by normalizing the images to have a mean close to zero and equal variance, the feature values are distributed evenly which makes it easier for the optimizer to find a good solution.

I initialized the weights using `tf.truncated_normal`, which generates normally distributed values with a mean of `mu` and standard deviation of `sigma`. It also drops any values that are more than 2 standard deviations from the mean and re-calculates them. This is important for training as extreme weight values would affect the gradients during backpropagation, and therefore affect learning. 

The model I used was based on the [LeNet architecture](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf) proposed by Yann LeCun. This CNN architecture is five-layer network with two convolutional layers and three fully connected layers. For my architecture, I modified the LeNet and used three convolutional layers and two fully connected layers and added regularized the network by adding dropout before the final layer.

### Testing on New Images

### Visualizing the Layers of the CNN

### Future Work
