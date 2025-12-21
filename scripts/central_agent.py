#!/usr/bin/env python3

import rospy
import collections
from std_msgs.msg import String

class CentralAgent:
    def __init__(self):
        rospy.init_node('central_agent')
        
        # Central Identifier
        self.node_id = "CENTRAL_TOWER"
        
        # Communication Channels
        self.pub = rospy.Publisher('/central/comm', String, queue_size=20)
        self.sub = rospy.Subscriber('/comm/agents', String, self.callback)
        
        # Async Processing Queue
        self.ack_queue = collections.deque()
        
        rospy.loginfo(f"[{self.node_id}] Online. Waiting for drone fleet to deploy...")
        
        # Wait for drones to spawn (approx 15-20s for 18 drones)
        rospy.sleep(20.0)
        
        rospy.loginfo(f"[{self.node_id}] Fleet Deployed. Starting 3-Way Handshake (Period: 10s)...")
        
        # Periodic "SYN" Broadcast (Step 1)
        self.timer = rospy.Timer(rospy.Duration(10.0), self.broadcast_hello)
        
        # Async ACK Processor (10Hz)
        self.process_timer = rospy.Timer(rospy.Duration(0.1), self.process_ack_queue)

    def broadcast_hello(self, event):
        """Step 1: Broadcast HELLO (SYN) to all units"""
        msg = "HELLO"
        rospy.loginfo(f"[{self.node_id}] >>> Broadcasting: {msg}")
        self.pub.publish(msg)

    def process_ack_queue(self, event):
        """Process one pending ACK per tick"""
        if self.ack_queue:
            sender_id = self.ack_queue.popleft()
            
            # Step 3: Send ACK (Async)
            ack_msg = f"TOWER_ACK_{sender_id}"
            self.pub.publish(ack_msg)
            rospy.loginfo(f"[{self.node_id}] Received HI from {sender_id}. ACK sent.")

    def callback(self, msg):
        """Handle responses from Agents"""
        data = msg.data
        
        # Step 2 Receive: Listen for HI (SYN-ACK)
        if "AGENT_HI" in data:
            # Parse sender ID
            try:
                # Format: AGENT_HI_<ID>
                sender_id = data.split("_HI_")[1]
                
                # Queue it for processing
                self.ack_queue.append(sender_id)
                
            except IndexError:
                pass

if __name__ == '__main__':
    try:
        CentralAgent()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
