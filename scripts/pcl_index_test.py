#!/usr/bin/env python
import rospy

from sensor_msgs.msg import PointCloud2
import matplotlib.pyplot as plt
import sensor_msgs.point_cloud2 as pc2
import torch



class PclTestNode:
    def read_pointcloud (self, pointcloud: PointCloud2) -> torch.Tensor:
        pcl_tensor = torch.empty ([pointcloud.width * pointcloud.height, 3], dtype=torch.float32)

        for i, p in enumerate (pc2.read_points(pointcloud, skip_nans=True, field_names=("x", "y", "z"))):
            for j in range(3):
                pcl_tensor[i][j] = p[j]
        
        return pcl_tensor

    def compare (self, old_pcl: torch.Tensor, new_pcl: torch.Tensor) -> torch.Tensor:
        return (new_pcl - old_pcl).norm (2,1)
    
    def plot(self, dist: torch.Tensor):
        self.fig.clear ()
        
        plt.hist (dist)
        plt.xlabel ("pcl index")
        plt.ylabel ("norm dist")
        plt.grid ()
        plt.draw ()


    def callback (self, pointcloud):
        if (self.old_pcl is None):
            self.old_pcl = self.read_pointcloud (pointcloud)
        else:
            new_pcl = self.read_pointcloud (pointcloud)
            dist = self.compare (self.old_pcl, new_pcl)
            self.plot (dist)

    def __init__(self):
        self.old_pcl = None
        rospy.init_node('listener', anonymous=True)
        rospy.Subscriber("/slife/debug/output_3", PointCloud2, self.callback)

        self.fig = plt.figure ()
        plt.show ()

    def spin (self):
        rospy.spin()

if __name__ == "__main__":
    ptn = PclTestNode ()
    ptn.spin ()