#!/usr/bin/env python3

import rospy
import threading
import random
from std_msgs.msg import String

class UGVCommManager:
    def __init__(self):
        rospy.init_node('ugv_comm_manager')
        
        self.ugvs = [VirtualUGV("UGV_1", self.pub_msg), VirtualUGV("UGV_2", self.pub_msg)]
        
        self.pub = rospy.Publisher('/comm/agents', String, queue_size=10)
        self.sub = rospy.Subscriber('/central/comm', String, self.callback)
        
        rospy.loginfo("[UGVNet] UGV Communication Channels Open.")

    def pub_msg(self, msg):
        self.pub.publish(msg)

    def callback(self, msg):
        cmd = msg.data
        
        if cmd == "HELLO":
            for ugv in self.ugvs:
                ugv.respond_to_hello()
                
        elif "TOWER_ACK" in cmd:
            try:
                target_id = cmd.split("_ACK_")[1]
                for ugv in self.ugvs:
                    if ugv.name == target_id:
                        ugv.receive_ack()
            except:
                pass

class VirtualUGV:
    def __init__(self, name, pub_func):
        self.name = name
        self.pub_func = pub_func
        self.connected = False

    def respond_to_hello(self):
        delay = random.uniform(0.1, 1.5)
        def send():
            rospy.sleep(delay)
            rospy.loginfo(f"[{self.name}] Heard HELLO. Sending HI...")
            msg = f"AGENT_HI_{self.name}"
            self.pub_func(msg)
        threading.Thread(target=send).start()

    def receive_ack(self):
        if not self.connected:
            rospy.loginfo(f"[{self.name}] Connection Established. (ACK Received)")
            self.connected = True

if __name__ == '__main__':
    try:
        UGVCommManager()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
