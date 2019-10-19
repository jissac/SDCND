#!/usr/bin/env python

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, TwistStamped
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree
from std_msgs.msg import Int32, Bool
import yaml
import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.
'''

LOOKAHEAD_WPS = 50 # Number of waypoints we will publish.
MAX_DECEL = 0.5

class WaypointUpdater(object):
    def __init__(self):
        # Initialize ros node
        rospy.init_node('waypoint_updater',log_level=rospy.DEBUG)

        # Subscribers
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        rospy.Subscriber('/obstacle_waypoint', Int32, self.obstacle_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.velocity_cb)
        
        # Publishers
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # Member variables
        self.pose = None
        self.base_waypoints = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.current_vel = 0.0
        self.max_accel = 1.0 # m/s - Maximum acceleration rate to keep jerk below 10m/s^3
        self.max_decel = 4.0 # m/s - Maximum deceleration rate to keep jerk below 10m/s^3
        self.stopline_wp_idx = -1
        self.light_state = -1

        self.loop()

    def loop(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.pose and self.base_waypoints:#self.base_lane:
                self.publish_waypoints()
            rate.sleep()

    def get_closest_waypoint_idx(self):
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        
        # Each waypoint in the waypoint_tree is stored as [position, index]
        # The query() function will return the closest waypoint to [x, y], and
        #  the "1" value specifies to return only one item. We are then taking
        #  only the index ([1])
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]
        
        # Check if the closest waypoint is ahead of, or behind the vehicle
        # We are looking for the waypoint in front of the vehicle here
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx-1]
        
        # Equation for hyperplane through closest_coords
        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([x, y])
        
        val = np.dot(cl_vect-prev_vect, pos_vect-cl_vect)
        # Alternatively, you can take the orientation of the vehicle, and the 
        #  orientation of a vector from the previous waypoint to the current
        #  waypoint and compare them to determine if they are facing in the 
        #  same direction.
                
        if val > 0:
            # Waypoint is behind the vehicle, so increment index forward by one
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)
        #print(closest_idx)
        return closest_idx

    def publish_waypoints(self):
        final_lane = self.generate_lane()
        self.final_waypoints_pub.publish(final_lane)

        
    def generate_lane(self):
        lane = Lane()
        lane.header = self.base_waypoints.header
        closest_idx = self.get_closest_waypoint_idx()
        farthest_idx = closest_idx + LOOKAHEAD_WPS
        base_waypoints = self.base_waypoints.waypoints[closest_idx:farthest_idx]
        if self.stopline_wp_idx == -1 or (self.stopline_wp_idx > farthest_idx):
            lane.waypoints = base_waypoints[0:LOOKAHEAD_WPS-25]                
        else: 
            lane.waypoints = self.decelerate_waypoints(base_waypoints,closest_idx)[0:LOOKAHEAD_WPS-25]
        return lane
    
    def decelerate_waypoints(self,waypoints, closest_idx):
        temp = []
        stop_idx = max(self.stopline_wp_idx - closest_idx - 5, 0)
        for i, wp in enumerate(waypoints):
            p = Waypoint()
            p.pose = wp.pose
            dist = self.distance(waypoints, i, stop_idx)
            vel = math.sqrt(2*MAX_DECEL*dist)
            if vel <1.:
                vel = 0.
            p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
            temp.append(p)
        return temp
    
    def pose_cb(self, msg):
        self.pose = msg
    
    def velocity_cb(self, msg):
        self.current_vel = msg.twist.linear.x

    def waypoints_cb(self, waypoints):
        self.base_waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        self.stopline_wp_idx = msg.data

    def obstacle_cb(self, msg):
        self.obstacle_wp_idx = msg.data

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
