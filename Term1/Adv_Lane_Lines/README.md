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
[image3]: ./img/output_images/test_binary.jpg "test_binary"
[image4]: ./img/output_images/test_roi.jpg "test_roi"
[image5]: ./img/output_images/test_binary_warped.jpg "test_binary_warped"
[image6]: ./img/output_images/test_histogram.jpg "test_histogram"
[image7]: ./img/output_images/test_polynomial.jpg "test_polynomial"
[image8]: ./img/output_images/test_final.jpg "test_final"
[video1]: ./project_video.mp4 "Video"

### Camera Calibration

Cameras look at a 3D object in the real world and translate that into 2D - this process changes what the shape and size of the 3D object appear to be. Therefore, the first step in analyzing camera images should be to undistort the recorded image. There are two primary types of distortion: **radial** and **tangential**. Radial distortion refers to when lines or objects appear more or less curved than they actually are. Tangential distortion refers to when a camera's lens is not aligned perfectly parallel to the imaging plane, where the camera film or sensor is, which results in a tilted image. 
  
**Map distorted points to undistorted points**       
We can correct for these distortion errors by calibrating with pictures of known objects. An image of a chessboard is great for calibration because its regular high contrast pattern makes it easy to detect automatically. So if we use our camera to take multiple images of a chessboard pattern against a flat surface, we can detect distortion by comparing the apparent size and shape of the squares in these images to a standard image of a chessboard. We'll create a transform that maps the distorted points to the undistorted points which will then allow us to undistort any image, as shown below.

![alt text][image1]

Using OpenCV and Python, I computed the camera matrix and distortion coefficients. I created an array called `objpoints`, which holds the (x,y,z) coordinates of the 'ground-truth' chessboard coordinates for an undistorted 3D image (I assume the chessboard is fixed on the (x,y) plane at a fixed distance z=0), and an array called `imgpoints`, which holds the (x,y) coordinates of the images in the calibration image plane. I found the corners of each calibration image using OpenCV, stored them in `imgpoints`, and mapped those points to the ground-truths contained in `objpoints`. I used these mapped values to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function and then applied the distortion correction to a test image using the `cv2.undistort()` function.

The code for this step is contained in the `calibrate_camera()` and `undistort()` functions on lines 15 through 64 in the file `lane-utils.py`. An example of an original test image with its undistorted image is shown below:

![alt text][image2]


### Gradient and Color Thresholding
Our eyes can easily detect which pixels belong to lanes and which do not. However, isolating these pixels for a computer to 'see' requires some thresholding - or image segmentation. The goal of this step is to produce a binary image where the lane line pixels are clearly visible; this is accomplished by using gradient and color thresholding. 

I do gradient thresholding by using the Sobel operator to compute the *x* and *y* derivatives of the image, thereby finding the vertical and horizontal edges in the image, respectively. I then compute the magnitude of the gradient by combining both *x* and *y* derivatives. Furthermore, in the case of lane lines, we're only interested in edges of a particular orientation. By finding the direction of the gradient (the inverse tangent (arctangent) of the *y* gradient divided by the *x* gradient) we can further isolate the lane lines and remove other noisy edges in the image.

Color spaces are important in an image as they convey valuable information not captured by gradients alone, such as yellow lane lines. I do color thresholding by isolating the S channel in the HLS (hue, lightness, saturation) color space. I found that the S channel most clearly isolates the lane lines out of the various color channels. 

Finally, I combine the color and gradient thresholds to arrive at a combined binary image. The code for this step is contained in the `color_gradient_thresh()` function on lines 66 through 121 in the file `lane-utils.py`. An example of the output binary image is shown below:

![alt text][image3]

### Perspective Transform
A perspective transform applies a 'birds-eye' view to the thresholded binary image by mapping the points in a given image to different, desired, image points with a new perspective. This is helpful when calculating lane curvature. 

First, I identified four source points for the perspective transform. I assumed the road is a flat plane and manually found the points, which form a trapezodial shape that would represent a rectangle when looking down on the road from above. The region of interest is shown below.

![alt text][image4]

The code for this step is contained in the `perspective_transform()` function on lines 123 through 150 in the file `lane-utils.py`. An example of the transformed image is shown below:

![alt text][image5]

### Lane Line Fitting
The final step in the pipeline is to map the isolated lane line pixels to left and right lane line polynomials and determine the lane curvature. I found where the left and right lane lines start by plotting a histogram of where the binary activations occur across the lower half of the image. By adding up the pixel values along each column in the image, I was able to find the two peaks where the base of the lane lines occur, as shown in the image below.

![alt text][image6]

After finding the starting points of the lane lines, I used sliding windows moving upward in the image (farther along the road) to determine where the lane lines go and fit polynomials to the left and right lanes, as shown in the image below.

![alt text][image7]

After finding and fitting a polynomial to the pixel positions, I computed the radius of curvature of the fit using the following formula: `f(y)=Ay^2+By+C`. The curvature of a given curve at a particular point is the curvature of the approximating circle at that point. Since the curvature depends on the radius, the smaller the radius, the greater the curvature (and vice versa); the radius of curvature *R* is the inverse of the curvature *K*. After finding the radius of curvature, I converted the number from pixel space to real-world space. 

The final segmented image with the overlayed lane information is shown below. The code for this step is found in the functions `hist()`, `find_lane_pixels()`, `fit_polynomial()`, `measure_curvature_pixels_meters()`, `lane_offset()`, and `draw_fill_lanes()` on lines 142 throught 330 in the file `lane-utils.py`.

![alt text][image8]

### Video Pipeline
After completing the pipeline for image frames, the final part of the project is to extend the pipeline to work on video (sequential image frames). I found that averaging the lane line predictions over the previous 10 frames gave smoother results. Furthermore, instead of repeating the lane-line search from scratch every frame, after the first detection I did a highly targeted search in a margin around the previous line position, as shown below (the green shaded area shows where I searched for the lines).

** add green search image **

The code for this step is found in the functions `hist()`, `find_lane_pixels()`, `fit_polynomial()`, `measure_curvature_pixels_meters()`, `lane_offset()`, and `draw_fill_lanes()` on lines 142 throught 330 in the file `lane-utils.py`.

### Lessons Learned and Future Work
I learned a lot during this challenging project - especially the end-to-end work needed to design an effective lane finding algorithm. There were a lot of parts to fit together in order to copmlete the project, from camera calibration to correctly mapping out the lanes. I spent a lot of time
For future work, I can improve on the algorithm to work robustly on the challenge video. Due to the intense sun glare and sharp turns in the video, my lane finding algorithm struggled to consistently map out the lanes. 
