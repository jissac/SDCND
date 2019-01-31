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
* Visualizing the CNN's hidden layers 

![](adv_lane_line.gif)

[//]: # (Image References)

[image1]: ./img/output_images/cal_undist.jpg "cal_undist"
[image2]: ./img/output_images/test_undist.jpg "test_undist"

### Dataset Exploration
The [German traffic dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) is a set of 50,000 images, each image containing one traffic sign each, with a total of 43 classes. 
![alt text][image1]

### Design and Test Model Architecture
I started by preprocessing the training and validation sets in order to make it easier for the CNN to learn during training: by normalizing the images to have a mean close to zero and equal variance, the feature values are distributed evenly which makes it easier for the optimizer to find a good solution.

I initialized the weights using `tf.truncated_normal`, which generates normally distributed values with a mean of `mu` and standard deviation of `sigma`. It also drops any values that are more than 2 standard deviations from the mean and re-calculates them. This is important for training becuase if the values 

The generated values follow a normal distribution with specified mean and standard deviation, except that values whose magnitude is more than 2 standard deviations from the mean are dropped and re-picked.


If we didn't scale our input training vectors, the ranges of our distributions of feature values would likely be different for each feature, and thus the learning rate would cause corrections in each dimension that would differ (proportionally speaking) from one another. We might be over compensating a correction in one weight dimension while undercompensating in another.

This is non-ideal as we might find ourselves in a oscillating (unable to center onto a better maxima in cost(weights) space) state or in a slow moving (traveling too slow to get to a better maxima) state.

It is of course possible to have a per-weight learning rate, but it's yet more hyperparameters to introduce into an already complicated network that we'd also have to optimize to find. Generally learning rates are scalars.

Thus we try to normalize images before using them as input into NN (or any gradient based) algorithm.
### Testing on New Images

### Visualizing the Layers of the CNN

### Future Work
