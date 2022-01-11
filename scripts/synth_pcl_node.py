#!/usr/bin/env python

import rospy
import torch
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import TransformStamped
import sensor_msgs.point_cloud2 as pc2
from roslib import message


synth_pcl_topic = "/camera/depth/pointcloud"
ground_truth_topic = "/vicon/depth_camera/depth_camera"
points_count = 20

class Pose:
    def __init__ (self, t, q):
        self.t = torch.tensor (t,dtype=torch.float)
        self.q = torch.tensor (q,dtype=torch.float)

ground_truth = Pose([0.1,0.0,0.0],[0.0499792, 0, 0, 0.9987503]) 
identity = Pose([0,0,0], [0,0,0,1])

def quat2rot (quat):
    x = quat[0]
    y = quat[1]
    z = quat[2]
    w = quat[3]

    x2 = x * x
    y2 = y * y
    z2 = z * z
    w2 = w * w

    xy = x * y
    zw = z * w
    xz = x * z
    yw = y * w
    yz = y * z
    xw = x * w

    matrix = torch.eye(3)

    matrix[0, 0] = x2 - y2 - z2 + w2
    matrix[1, 0] = 2 * (xy + zw)
    matrix[2, 0] = 2 * (xz - yw)

    matrix[0, 1] = 2 * (xy - zw)
    matrix[1, 1] = - x2 + y2 - z2 + w2
    matrix[2, 1] = 2 * (yz + xw)

    matrix[0, 2] = 2 * (xz + yw)
    matrix[1, 2] = 2 * (yz - xw)
    matrix[2, 2] = - x2 - y2 + z2 + w2

    return matrix

class SynthPclNode:
    def __init__ (self):
        rospy.init_node ("synth_pcl")
        self.pcl_pub = rospy.Publisher (synth_pcl_topic, PointCloud2, queue_size=1)
        self.ground_truth_pub = rospy.Publisher (ground_truth_topic, TransformStamped, queue_size=1)
    
    def create_pcl (self):
        self.pcl = torch.rand ([points_count, 3], dtype=torch.float)
        #self.pcl = torch.tensor ([[-0.1,0.1,0],[0.1,0,0],[0,-0.1,0]], dtype=torch.float32)

    def transform (self, pcl: torch.Tensor, pose: Pose):
        pos = pose.t.unsqueeze (0)
        rotm = quat2rot(pose.q)

        return rotm.mm (pcl.transpose(0,1)).transpose(0,1) + pos

    def send_pcl (self, do_transform: bool):
        print ("pcl")
        if (do_transform):
            self.pcl = self.transform (self.pcl, ground_truth)
            
        self.publish (self.pcl)
    
    def send_ground_truth (self, do_transform: bool):
        print ("gt")
        ground_truth_transform = None
        if (do_transform):        
            ground_truth_transform = ground_truth
        else:
            ground_truth_transform = identity
            
        ground_truth_msg = TransformStamped ()
        ground_truth_msg.transform.translation.x = ground_truth_transform.t[0]
        ground_truth_msg.transform.translation.y = ground_truth_transform.t[1]
        ground_truth_msg.transform.translation.z = ground_truth_transform.t[2]
        
        ground_truth_msg.transform.rotation.x = ground_truth_transform.q[0]
        ground_truth_msg.transform.rotation.y = ground_truth_transform.q[1]
        ground_truth_msg.transform.rotation.z = ground_truth_transform.q[2]
        ground_truth_msg.transform.rotation.w = ground_truth_transform.q[3]
        
        self.ground_truth_pub.publish (ground_truth_msg)


    def publish (self, pcl: torch.Tensor):
        pclMsg = PointCloud2 ()
        pclMsg.width = points_count
        pclMsg.height = 1
        pclMsg = pc2.create_cloud_xyz32 (pclMsg.header,pcl.numpy().tolist())
        self.pcl_pub.publish (pclMsg)

    def spin (self):
        rospy.sleep (3)
        for i in range(10):
            self.create_pcl ()
            self.send_ground_truth(False)
            self.send_pcl (False)
            self.send_ground_truth(False)
            self.send_pcl (True)
        # rospy.spin ()

if __name__ == "__main__":
    try:
        spn = SynthPclNode ()
        spn.spin ()

    except rospy.ROSInterruptException:
        pass