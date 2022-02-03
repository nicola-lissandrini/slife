#!/usr/bin/env python


import rospy
import numpy as np
from std_msgs.msg import Header

class ClockDiffNode:
    def __init__ (self):
        rospy.init_node ("clock_diff")
        self.clock_sub = rospy.Subscriber ("/clock", Header, self.clock_test, queue_size=1)

    def clock_test(self, msg: Header):
        this_time = rospy.Time.now ()
        other_time = msg.stamp
        diff_time = other_time - this_time
        print ("Ros master vs client diff {}ms".format (diff_time.to_sec()*1000))


    def spin(self):
        rospy.spin ()
if __name__ == "__main__":
    try:
        cdn = ClockDiffNode ()
        cdn.spin ()
    except rospy.ROSInterruptException:
        pass