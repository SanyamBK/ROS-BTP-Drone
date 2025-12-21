#!/usr/bin/env python3

import rospy
import threading
import random
from std_msgs.msg import String

class DroneCommManager:
    def __init__(self):
        rospy.init_node('drone_comm_manager')
        
        self.num_drones = 18
        self.drones = []
        
        self.pub = rospy.Publisher('/comm/agents', String, queue_size=20)
        self.sub = rospy.Subscriber('/central/comm', String, self.callback)
        
        rospy.loginfo(f"[DroneNet] Initializing fleet of {self.num_drones} drones...")
        
        # Create virtual drone states
        for i in range(self.num_drones):
            self.drones.append(VirtualDrone(i, self.pub))

    def callback(self, msg):
        """Handle messages from Central Tower"""
        cmd = msg.data
        
        if cmd == "HELLO":
            # Step 1 Receive: Tower initiates handshake
            # Propagate to all drones to respond
            for drone in self.drones:
                drone.respond_to_hello()
                
        elif "TOWER_ACK" in cmd:
            # Step 3 Receive: Tower Acknowledged us
            try:
                target_id = cmd.split("_ACK_")[1]
                # Notify specific drone
                if target_id.startswith("DRONE_"):
                    idx = int(target_id.split("_")[1])
                    if 0 <= idx < self.num_drones:
                        self.drones[idx].receive_ack()
            except:
                pass

class VirtualDrone:
    def __init__(self, drone_id, pub):
        self.id = drone_id
        self.name = f"DRONE_{drone_id}"
        self.pub = pub
        self.connected = False

    def respond_to_hello(self):
        """Step 2: Send HI (SYN-ACK)"""
        # Add small random delay to prevent network congestion/collision in sim
        delay = random.uniform(0.1, 2.0)
        
        def send():
            rospy.sleep(delay)
            # LOG THE MIDDLE PART
            rospy.loginfo(f"[{self.name}] Heard HELLO. Sending HI...")
            msg = f"AGENT_HI_{self.name}"
            self.pub.publish(msg)
        
        threading.Thread(target=send).start()

    def receive_ack(self):
        """Step 3 Complete: Connection Established"""
        if not self.connected:
            rospy.loginfo(f"[{self.name}] Connection Established! (ACK Received)")
            self.connected = True
        else:
            # Keep alive
            pass

if __name__ == '__main__':
    try:
        DroneCommManager()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
