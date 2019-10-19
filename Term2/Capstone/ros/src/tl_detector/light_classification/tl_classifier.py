from styx_msgs.msg import TrafficLight
import rospy
import cv2
import numpy as np
import tensorflow as tf
import os
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class TLClassifier(object):
    def __init__(self):
        pass
    
    def get_classification(self, image):
        """Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        light = TrafficLight.UNKNOWN
        
        # HSV allows count color within hue range
        hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Color definitions (Numpy defines colors as BGR)
        RED1_lower = np.array([17, 15, 100], dtype=np.uint8)
        RED1_upper = np.array([15, 56, 200], np.uint8)
        RED2_lower = np.array([160, 100, 100], np.uint8)
        RED2_upper = np.array([179, 255, 255], np.uint8)
        
        YELLOW_lower = np.array([28, 100, 100], np.uint8)
        YELLOW_upper = np.array([47, 255, 255], np.uint8)
        
        GREEN_lower = np.array([52, 146, 62], np.uint8)
        GREEN_upper = np.array([85, 255, 108], np.uint8)

        # Define masks 
        red_mask1 = cv2.inRange(hsv_img, RED1_lower, RED1_upper)
        red_mask2 = cv2.inRange(hsv_img, RED2_lower, RED2_upper)
        red1_count = cv2.countNonZero(red_mask1)
        red2_count = cv2.countNonZero(red_mask2)
        
        green_mask = cv2.inRange(hsv_img, GREEN_lower, GREEN_upper)
        green_count = cv2.countNonZero(green_mask)
        
        yellow_mask = cv2.inRange(hsv_img, YELLOW_lower, YELLOW_upper)
        yellow_count = cv2.countNonZero(yellow_mask)
        
        # Determine if there are more red or green counts in the image
        if (((red1_count + red2_count) > (green_count)) | ((red1_count + red2_count) > (yellow_count))):
            color = 'red'
            light = TrafficLight.RED
        else:
            color = 'green'
            light = TrafficLight.GREEN
   
        #print("The light is: {} ({})".format(color,light))
        return light   