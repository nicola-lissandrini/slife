#!/usr/bin/env python

from cProfile import label
from os import curdir
import signal
import matplotlib.pyplot as plt

import rospy
import numpy as np
from std_msgs.msg import Float32MultiArray

titles = ["estimate", "history", "error_history", "final_error", "relative_ground_truth","estimate_world", "processed_pointcloud", "misc"]

def handle_close (fig):
    quit ()

def signal_handler (sig, frame):
    plt.close ()

class ScopeNode:
    def __init__ (self):
        rospy.init_node ("scope")

        self.scope_topic = rospy.get_param("~topic")
        self.ymin = rospy.get_param("~y_min")
        self.ymax = rospy.get_param("~y_max")
        self.dims = rospy.get_param("~dims", 3)
        self.hist = rospy.get_param("~hist", False)
        self.record = rospy.get_param("~record", True)
        self.tensor_sub = rospy.Subscriber (self.scope_topic, Float32MultiArray, self.tensor_callback, queue_size=1)
        self.plot = None
        self.fig = plt.figure ()
        self.fig.canvas.mpl_connect ('close_event', handle_close)
        plt.show ()

    def update (self, tensor):
        if (self.plot is None):
            self.plot = tensor
        else:
            if (self.record):
                self.plot = np.concatenate ((self.plot, tensor))
            else:
                self.plot = tensor

        self.draw ()

    def draw (self):
        self.fig.clear ()

        plt.clf ()
        
        if (self.hist):
            # NOT WORKING
            plt.hist (self.plot.flatten ())
        else:
            for i in range(self.dims):
                plt.plot (self.plot[:,i], label="dim {}".format (i))
            plt.ylim ([self.ymin, self.ymax])
        plt.title (titles[self.type])
        plt.legend ()
        plt.grid ()
        plt.draw ()

    def tensor_callback (self, tensor_msg: Float32MultiArray):
        sizes = []

        for curr_dim in tensor_msg.layout.dim:
            sizes.append (curr_dim.size)

        self.type = int (tensor_msg.data[0])
        tensor = np.array (tensor_msg.data[tensor_msg.layout.data_offset:]).reshape (sizes).reshape (1,-1)
        self.update (tensor)

if __name__ == "__main__":
    try:
        signal.signal (signal.SIGINT, signal_handler)
        sn = ScopeNode ()
        sn.spin ()
    except rospy.ROSInterruptException:
        pass