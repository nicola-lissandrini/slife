#!/usr/bin/env python

import signal
from matplotlib.pyplot import grid
import matplotlib.pyplot as plt

import rospy
import numpy as np
from std_msgs.msg import Float32MultiArray
import torch

history_topic = "/slife/debug/history"

class Pose:
    def __init__ (self, t, q):
        self.t = torch.tensor (t,dtype=torch.float)
        self.q = torch.tensor (q,dtype=torch.float)

ground_truth = Pose([0.5,0.0,0.0],[0,0,0,1])


def handle_close (fig):
    quit ()

def signal_handler (sig, frame):
    plt.close ()

class DisplayHistoryNode:
    def tensor_callback (self, tensor_msg):
        self.sizes = []
        for curr_dim in tensor_msg.layout.dim:
            self.sizes.append (curr_dim.size)
            
        self.tensor =torch.from_numpy (np.array (tensor_msg.data).reshape (self.sizes))
        self.draw ()

    def process_data (self, tensor):
        t = tensor[:,0:3]
        q = tensor[:,3:]

        error = (t - ground_truth.t).norm (2,1)
        
        return error

    def draw (self):
        if (self.tensor is None):
            return

        self.fig.clear ()

        error = self.process_data (self.tensor)
        
        plt.plot (error)
        plt.grid ()
        plt.draw ()

    def __init__ (self):
        rospy.init_node ("display_history")
        self.tensor_sub = rospy.Subscriber (history_topic, Float32MultiArray, self.tensor_callback, queue_size=1)
        self.tensor = None
        
        self.fig = plt.figure ()
        self.fig.canvas.mpl_connect ('close_event', handle_close)
        plt.show ()

if __name__ == "__main__":
    try:
        signal.signal (signal.SIGINT, signal_handler)
        dhn = DisplayHistoryNode ()
        dhn.spin ()
    except rospy.ROSInterruptException:
        pass