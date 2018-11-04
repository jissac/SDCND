# **Finding Lane Lines on the Road** 

In this first Udacity Self-Driving Car Nanodegree project, I was tasked with finding lane lines on images using computer
vision techniques. 
After finding lanes on images, I extended the pipeline to work on video streams. 


## Pipeline

The series of steps I followed to find lane lines is as follows:
1. Convert the image to grayscale
2. Apply a Gaussian blur to the image
3. Detect edges in the image using the Canny algorithm
4. Create a region-of-interest that masks only the portions of the image that contain lane lines
5. Use the Hough transformation to find line segments based on the edges
6. Extrapolate the disjointed left and right line segments to form cohesive left and right lane markings

## Drawing lane-lines
Drawing a single, solid line for both left and right lane lines was a challenge.  


