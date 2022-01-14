#!/usr/bin/env python

import signal
from matplotlib.pyplot import grid
import matplotlib.pyplot as plt

import rospy
import numpy as np
from std_msgs.msg import Float32MultiArray
import torch

history_topic = "/slife/debug/ground_truth_error"

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
        error_t = tensor[:,0:3]
        error_q = tensor[:,3:]
        print (tensor.shape)
        return error_t, error_q

    def draw (self):
        if (self.tensor is None):
            return

        self.fig.clear ()

        error_t, error_q = self.process_data (self.tensor)
        
        #plt.scatter (self.tensor[:,0], self.tensor[:,1])
        plt.clf()
        plt.plot(error_t, label="translation")
        #plt.plot(error_q, label="rotation")
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

    def spin (self):
        rospy.spin()

if __name__ == "__main__":
    try:
        signal.signal (signal.SIGINT, signal_handler)
        dhn = DisplayHistoryNode ()
        dhn.spin ()
    except rospy.ROSInterruptException:
        pass