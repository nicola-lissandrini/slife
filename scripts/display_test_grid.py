#!/usr/bin/env python

import signal
from matplotlib.pyplot import grid
import rospy
import numpy as np
from std_msgs.msg import Float32MultiArray
import matplotlib.pyplot as plt
import math

grid_topic = "/slife/debug/grid"

# Type enum
TEST_NONE = 0
TEST_LANDSCAPE_VALUES = 1
TEST_LANDSCAPE_GRADIENT = 2
TEST_COST_VALUES = 3
TEST_COST_GRADIENT = 4

def handle_close (fig):
    quit ()

def signal_handler (sig, frame):
    plt.close ()

class DisplayNode:
    def test_grid_callback (self, grid_msg):
        self.ranges = grid_msg.data[0:3]
        self.type = int (round (grid_msg.data[3]))

        self.sizes = []
        for curr_dim in grid_msg.layout.dim:
            self.sizes.append (curr_dim.size)

        self.values = np.array (grid_msg.data[4 :]).reshape (self.sizes)
        print (self.values)
        self.draw ()

    def draw (self):
        if (self.values is None):
            return
        
        self.fig.clear ()
        xy_range = np.arange (self.ranges[0], self.ranges[1], self.ranges[2])
        x, y = np.meshgrid (xy_range, xy_range)

        if (self.type == TEST_LANDSCAPE_VALUES):
            im = plt.pcolormesh (y, x, self.values, edgecolors="none", antialiased=True, vmin=0, vmax=1)
            self.fig.colorbar (im)
        elif (self.type == TEST_COST_VALUES):
            im = plt.pcolormesh (y, x, self.values, edgecolors="none", antialiased=True)
            self.fig.colorbar (im)
        else:
            print (self.values[:,0])
            sqrtsize = int (math.sqrt (self.sizes[0]))
            #im = plt.pcolormesh (y, x, self.values[:,0].reshape (sqrtsize, sqrtsize), edgecolors="none", antialiased=True, cmap="RdBu")
            #self.fig.colorbar (im)
            plt.quiver (y,x,self.values[:,0], self.values[:,1], minshaft=0.1)
        
        plt.grid ()
        plt.gca().set_xlim (xy_range[0], xy_range[-1])
        plt.gca().set_ylim (xy_range[0], xy_range[-1])
        plt.gca().set_aspect('equal', adjustable='box')
        plt.draw ()

    def __init__ (self):
        rospy.init_node ("display_measures")
        self.map_sub = rospy.Subscriber (grid_topic, Float32MultiArray, self.test_grid_callback, queue_size=1)
        
        self.fig = plt.figure ()        
        self.fig.canvas.mpl_connect('close_event', handle_close)
        self.values = None 
        plt.show ()

    def spin (self):
        rospy.spin ()

if __name__ == "__main__":
    try:
        signal.signal(signal.SIGINT, signal_handler)
        dn = DisplayNode ()
        dn.spin ()
    except rospy.ROSInterruptException:
        pass