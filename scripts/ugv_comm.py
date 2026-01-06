#!/usr/bin/env python3

import rospy
import threading
import random
import os
from datetime import datetime
from std_msgs.msg import String

class UGVCommManager:
    def __init__(self):
        rospy.init_node('ugv_comm_manager')

        # Connection log file (shared with central agent)
        self.log_file = self._init_log_file()

        self.pub = rospy.Publisher('/comm/agents', String, queue_size=10)
        self.sub = rospy.Subscriber('/central/comm', String, self.callback)

        self.ugvs = [VirtualUGV("UGV_1", self.pub_msg, self._log_to_file), VirtualUGV("UGV_2", self.pub_msg, self._log_to_file)]
        
        rospy.loginfo("[UGVNet] UGV Communication Channels Open.")
        self._log_to_file("[UGVNet] UGV Communication Channels Open.")

    def pub_msg(self, msg):
        self.pub.publish(msg)

    def callback(self, msg):
        cmd = msg.data
        
        if cmd == "HELLO":
            for ugv in self.ugvs:
                ugv.respond_to_hello()
            self._log_to_file("[UGVNet] HELLO received from central")
                
        elif "TOWER_ACK" in cmd:
            try:
                target_id = cmd.split("_ACK_")[1]
                for ugv in self.ugvs:
                    if ugv.name == target_id:
                        ugv.receive_ack()
                        self._log_to_file(f"[{ugv.name}] ACK received from central")
            except:
                pass

    def _init_log_file(self):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        logs_dir = os.path.join(base_dir, 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        return os.path.join(logs_dir, 'connection_report.log')

    def _log_to_file(self, message: str):
        timestamp = datetime.now().isoformat()
        try:
            with open(self.log_file, 'a') as fh:
                fh.write(f"[{timestamp}] {message}\n")
        except Exception:
            pass

class VirtualUGV:
    def __init__(self, name, pub_func, log_func):
        self.name = name
        self.pub_func = pub_func
        self.log = log_func
        self.connected = False

    def respond_to_hello(self):
        delay = random.uniform(0.1, 1.5)
        def send():
            rospy.sleep(delay)
            rospy.loginfo(f"[{self.name}] Heard HELLO. Sending HI...")
            if self.log:
                self.log(f"[{self.name}] Heard HELLO. Sending HI...")
            msg = f"AGENT_HI_{self.name}"
            self.pub_func(msg)
        threading.Thread(target=send).start()

    def receive_ack(self):
        if not self.connected:
            rospy.loginfo(f"[{self.name}] Connection Established. (ACK Received)")
            if self.log:
                self.log(f"[{self.name}] Connection Established. (ACK Received)")
            self.connected = True

if __name__ == '__main__':
    try:
        UGVCommManager()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
