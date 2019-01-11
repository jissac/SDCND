'''
Define functions for lane-finding project
'''

import glob
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle


# functions
def calibrate_camera(calibration_images):
    '''
    Compute the camera calibration matrix and distortion coefficients given calibration images
    '''
    # number of inside corners
    nx = 9
    ny = 6

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space for an undistorted image
    imgpoints = [] # 2d points in calibration image plane

    # define x,y,z points for an undistorted 3D image from 
    # top left: (0,0,0) to bottom right: (nx-1,ny-1,0)
    objp = np.zeros((nx*ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)


    # Step through the list and search for chessboard corners
    for i,fname in enumerate(calibration_images):
        img = cv2.imread(fname)

        # Convert to grayscale
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        #print(ret,corners)
        #print('---------------------\n')

        # if corners found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
    
    # Read in an image and calibrate
    img = cv2.imread('./camera_cal/calibration4.jpg')
    img_size = (img.shape[1], img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
    
    return objpoints, imgpoints, ret, mtx, dist, rvecs, tvecs

def color_gradient_thresh(img,s_thresh=(170, 255)):
    '''
    Apply color and gradient thresholding to generate a binary image where the lane lines are clearly visible
    '''
    # color threshold
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    H = hls[:,:,0]
    L = hls[:,:,1]
    S = hls[:,:,2]
    s_binary = np.zeros_like(S)
    s_binary[(S >= s_thresh[0]) & (S <= s_thresh[1])] = 1
    
    # gradient threshold (sobel magnitude and direction)
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    kernel_size = 3 
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

    #sobelx
    sobel_kernel = 3
    sobelx = cv2.Sobel(blur_gray, cv2.CV_64F, 1, 0,ksize=sobel_kernel)
    abs_sobelx = np.absolute(sobelx)

    #sobely
    sobely = cv2.Sobel(blur_gray, cv2.CV_64F, 0, 1,ksize=sobel_kernel)
    abs_sobely = np.absolute(sobely)

    #sobel scaled and binary
    scaled_abs_sobelx = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    scaled_abs_sobely = np.uint8(255*abs_sobely/np.max(abs_sobely))
    gradx_binary = np.zeros_like(scaled_abs_sobelx)
    grady_binary = np.zeros_like(scaled_abs_sobely)
    sobel_thresh = (30,100)
    gradx_binary[(scaled_abs_sobelx >= sobel_thresh[0]) & (scaled_abs_sobelx <= sobel_thresh[1])] = 1
    grady_binary[(scaled_abs_sobely >= sobel_thresh[0]) & (scaled_abs_sobely <= sobel_thresh[1])] = 1
    
    # magnitude of sobelx and sobely
    absmag = abs_sobelx + abs_sobely
    scale_factor = np.max(absmag)/255 # Rescale to 8 bit (0-255)
    scaled_absmag = (cv2.GaussianBlur(absmag,(9, 9),0)/scale_factor).astype(np.uint8)
    mag_binary = np.zeros_like(scaled_absmag)
    mag_thresh = (30, 200)
    mag_binary[(scaled_absmag >= mag_thresh[0]) & (scaled_absmag <= mag_thresh[1])] = 1

    #direction of the gradient
    grad_dir_thresh=(0.7, 1.3)
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    dir_binary =  np.zeros_like(absgraddir)
    dir_binary[(absgraddir >= grad_dir_thresh[0]) & (absgraddir <= grad_dir_thresh[1])] = 1

    #combining sobelx,sobely,magnitude,direction,and color thresholds
    combined_output = np.zeros_like(dir_binary)
    combined_output[((gradx_binary == 1) & (grady_binary == 1)) | \
                    ((mag_binary == 1) & (dir_binary == 1)) | \
                    (s_binary == 1)] = 1
    
    return combined_output

def undistort(img,mtx,dist):
    '''
    Undistort image using camera calibration matrices
    '''
    undistorted = cv2.undistort(img, mtx, dist, None, mtx)
    
    return undistorted

def perspective_transform(img):
    '''
    
    '''
    top_left = (560,480)
    top_right = (760,480)
    bottom_right = (1150,720)
    bottom_left = (260,720)
    
    img_size = (image.shape[1], image.shape[0])
    offset = 320
    region_of_interest = np.array([[top_left,top_right, bottom_right, bottom_left]], dtype=np.float32)
    # destination points chosen so that warped lane lines appear parallel
    destination = np.float32([[offset, 0], [img_size[0]-offset, 0], 
                                     [img_size[0]-offset, img_size[1]], 
                                     [offset, img_size[1]]])
#     tl = [320,0]
#     tr = [920,0]
#     br = [920,720]
#     bl = [320,720]
#     destination = np.float32([tl,tr,br,bl])

    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(region_of_interest,destination)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, img_size)
    return warped, M

def curve_fit():
    
    return None
