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

ground_truth = Pose([0.1,0.0,0.0],[0.0499792, 0, 0, 0.9987503]) #[0.0871557, 0, 0, 0.9961947])


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
        #q = tensor[:,0:4]
        
        error_t = (t - ground_truth.t).norm (2,1)
        error_q = (q - ground_truth.q).norm (2,1)
        #error_q = t - ground_truth.t
        #error_q = None
        #error_t = None
        return error_t, error_q

    def draw (self):
        if (self.tensor is None):
            return

        self.fig.clear ()

        error_t, error_q = self.process_data (self.tensor)
        
        #plt.scatter (self.tensor[:,0], self.tensor[:,1])
        plt.plot(error_t, label="translation")
        plt.plot(error_q, label="rotation")
        plt.xlabel ("# iterations")
        plt.ylabel ("error norm (chordal metrics)")
        plt.legend ()
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