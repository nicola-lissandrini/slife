#!/usr/bin/env python

import rospy
import torch

synth_pcl_topic = "/camera/depth/pointcloud"
points_count = 10

class SynthPclNode:
    def __init__ (self):
        rospy.init_node ("synth_pcl")
        self.pcl_pub = rospy.Publisher (synth_pcl_topic, Float32MultiArray, queue_size=1)
    
    def get_pcl (self):
        return torch.rand ([points_count, 3])

    def send_pcl (self, do_transform):
        pcl = self.get_pcl ()

        if (do_transform):
            pcl = self.transform (pcl)

        self.publish (pcl)

    def spin (self):
        rospy.sleep (1)
        self.send_pcl (False)
        rospy.sleep (0.5)
        self.send_pcl (True)
        rospy.spin ()

if __name__ == "__main__":
    try:
        spn = SynthPclNode ()
        spn.spin ()

    except rospy.ROSInterruptException:
        pass