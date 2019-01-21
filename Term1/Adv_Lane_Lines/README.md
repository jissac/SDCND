## Advanced Lane Finding Project

### In this project, I wrote a software pipeline using traditional computer vision techniques to identify the lane boundaries from a car's front-facing camera video feed.

---

**Outline**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify the binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

An example of the final output is shown below:

** add image of final lane line output ***

[//]: # (Image References)

[image1]: ./img/output_images/cal_undist.jpg "cal_undist"
[image2]: ./img/output_images/test_undist.jpg "test_undist"
[image3]: ./img/output_images/warped.jpg "Warped"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

### Camera Calibration

Cameras look at a 3D object in the real world and translate that into 2D - this process changes what the shape and size of the 3D object appear to be. Therefore, the first step in analyzing camera images should be to undistort the recorded image. There are two primary types of distortion: **radial** and **tangential**. Radial distortion refers to when lines or objects appear more or less curved than they actually are. Tangential distortion refers to when a camera's lens is not aligned perfectly parallel to the imaging plane, where the camera film or sensor is, which results in a tilted image. 
  
**Map distorted points to undistorted points**       
We can correct for these distortion errors by calibrating with pictures of known objects. An image of a chessboard is great for calibration because its regular high contrast pattern makes it easy to detect automatically. So if we use our camera to take multiple images of a chessboard pattern against a flat surface, we can detect distortion by comparing the apparent size and shape of the squares in these images to a standard image of a chessboard. We'll create a transform that maps the distorted points to the undistorted points which will then allow us to undistort any image, as shown below.

![alt text][image1]

Using OpenCV and Python, I computed the camera matrix and distortion coefficients. I created a an array called `objpoints`, which holds the (x,y,z) coordinates of the 'ground-truth' chessboard coordinates for an undistorted 3D image (I assume the chessboard is fixed on the (x,y) plane at a fixed distance z=0), and an array called `imgpoints`, which holds the (x,y) coordinates of the images in the calibration image plane. I found the corners of each calibration image using OpenCV, stored them in `imgpoints`, and mapped those points to the ground-truths contained in `objpoints`. I used these mapped values to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function and then applied the distortion correction to a test image using the `cv2.undistort()` function.

The code for this step is contained in the `calibrate_camera()` and `undistort()` functions on lines 15 through 64 in the file `lane-utils.py`. An example of an original test image with its undistorted image is shown below:

![alt text][image2]


### Gradient and Color Thresholding
Our eyes can easily detect which pixels belong to lanes and which do not. However, isolating these pixels for a computer to 'see' requires some thresholding - or image segmentation. The goal of this step is to produce a binary image where the lane line pixels are clearly visible; this is accomplished by using gradient and color thresholding. 

I do gradient thresholding by using the Sobel operator to compute the *x* and *y* derivatives of the image, thereby finding the vertical and horizontal edges in the image, respectively. I then compute the magnitude of the gradient by combining both *x* and *y* derivatives. Furthermore, in the case of lane lines, we're only interested in edges of a particular orientation. By finding the direction of the gradient (the inverse tangent (arctangent) of the *y* gradient divided by the *x* gradient) we can further isolate the lane lines and remove other noisy edges in the image.

Color spaces are important in an image as they convey valuable information not captured by gradients alone, such as yellow lane lines. I do color thresholding by isolating the S channel in the HLS (hue, lightness, saturation) color space. I found that the S channel most clearly isolates the lane lines out of the various color channels. 

Finally, I combine the color and gradient thresholds to arrive at a combined binary image. The code for this step is contained in the `color_gradient_thresh()` function on lines 66 through 121 in the file `lane-utils.py`. An example of the output binary image is shown below:

** add binary output image **

### Perspective Transform
A perspective transform applies a 'birds-eye' view to the thresholded binary image by mapping the points in a given image to different, desired, image points with a new perspective. This is helpful when calculating lane curvature. 

First, I identified four source points for the perspective transform. I assumed the road is a flat plane and manually found the points, which form a trapezodial shape that would represent a rectangle when looking down on the road from above. The region of interest is shown below.

** add region of interest, trapezoidal image **

The code for this step is contained in the `perspective_transform()` function on lines 123 through 150 in the file `lane-utils.py`. An example of the transformed image is shown below:

** add warped image **

### Lane Line Fitting
The final step in the pipeline is to map the isolated lane line pixels to left and right lane line polynomials and determine the lane curvature. I found where the left and right lane lines start by plotting a histogram of where the binary activations occur across the lower half of the image. By adding up the pixel values along each column in the image, I was able to find the two peaks where the base of the lane lines occur, as shown in the image below.

** add histogram picture **

After finding the starting points of the lane lines, I used sliding windows moving upward in the image (further along the road) to determine where the lane lines go, as shown in the image below.

** add sliding window picture ** 

Furthermore, instead of repeating the lane-line search from scratch every frame, once you've found the lane line from the previous frame it's more efficient to do a highly targeted search in a margin around the previous line position, as shown below (the green shaded area shows where I searched for the lines this time).

** add green search image **

After finding and fitting a polynomial to the pixel positions, I computed the radius of curvature of the fit using the following formula: `f(y)=Ay^2+By+C`. The curvature of a given curve at a particular point is the curvature of the approximating circle at that point. Since the curvature depends on the radius, the smaller the radius, the greater the curvature (and vice versa); the radius of curvature *R* is the inverse of the curvature *K*. After finding the radius of curvature, I converted the number from pixel space to real-world space. 

The final segmented image with the overlayed lane information is shown below. The code for this step is found in the functions `code`, `code`, and `code` on lines x throught y in the file `lane-utils.py`.

### Video Pipeline
After completing the pipeline for a single image frame, the final part of the project is to extend the pipeline to work on video.


### Lessons Learned and Future Work

### References
Udacity 
https://www.intmath.com/applications-differentiation/8-radius-curvature.php

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image1]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image3]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
