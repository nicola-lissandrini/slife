#!/usr/bin/env python

import rospy
import ros_numpy
import torch
import sensor_msgs.point_cloud2 as pc2
import numpy as np
from sensor_msgs.msg import PointCloud2

import sensor_msgs.point_cloud2 as pc2

class SavePclNode:
    def pcl_callback (self, msg: PointCloud2):
        pc2pcl = pc2.read_points (msg)
        pcl_np = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg)
        pcl_tensor = torch.from_numpy (pcl_np.astype(np.float32))
        torch.save (pcl_tensor, "/home/nicola/real_pcl")
        print ("saved")
        
        
    def __init__(self):
        rospy.init_node ("save_pcl")
        self.pcl_sub = rospy.Subscriber ("/camera/depth/pointcloud", PointCloud2, self.pcl_callback, queue_size=1)
        print ("qua")

    def spin (self):
        rospy.spin ()


if __name__ == "__main__":
    try:
        spn = SavePclNode ()
        spn.spin ()
    except rospy.ROSInterruptException:
        pass