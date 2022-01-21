#!/usr/bin/env python

from os import curdir
import signal
import matplotlib.pyplot as plt

import rospy
import numpy as np
from std_msgs.msg import Float32MultiArray

scope_topic = "/slife/debug/output_1"

def handle_close (fig):
    quit ()

def signal_handler (sig, frame):
    plt.close ()

class ScopeNode:
    def __init__ (self):
        rospy.init_node ("scope")
        self.tensor_sub = rospy.Subscriber (scope_topic, Float32MultiArray, self.tensor_callback, queue_size=1)
        self.plot = None
        self.fig = plt.figure ()
        self.fig.canvas.mpl_connect ('close_event', handle_close)
        plt.show ()

    def update (self, tensor):
        if (self.plot is None):
            self.plot = tensor
        else:
            self.plot = np.concatenate ((self.plot, tensor))

        self.draw ()

    def draw (self):
        self.fig.clear ()

        plt.clf ()
        plt.plot (self.plot[:,0])
        plt.grid ()
        plt.draw ()

    def tensor_callback (self, tensor_msg: Float32MultiArray):
        sizes = []

        for curr_dim in tensor_msg.layout.dim:
            sizes.append (curr_dim.size)

        tensor = np.array (tensor_msg.data[tensor_msg.layout.data_offset:]).reshape (sizes).reshape (1,-1)
        self.update (tensor)

if __name__ == "__main__":
    try:
        signal.signal (signal.SIGINT, signal_handler)
        sn = ScopeNode ()
        sn.spin ()
    except rospy.ROSInterruptException:
        pass