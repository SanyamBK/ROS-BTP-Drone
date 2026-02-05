#!/usr/bin/env python3

import rospy
import collections
import os
from datetime import datetime
from std_msgs.msg import String, Bool

class CentralAgent:
    def __init__(self):
        rospy.init_node('central_agent')

        # Central Identifier
        self.node_id = "CENTRAL_TOWER"

        # Connection log file
        self.log_file = self._init_log_file()

        # Communication Channels
        self.pub = rospy.Publisher('/central/comm', String, queue_size=20)
        self.sub = rospy.Subscriber('/comm/agents', String, self.callback)

        # Mission end hook
        self.mission_sub = rospy.Subscriber('/mission_complete', Bool, self.on_mission_complete)

        # Async Processing Queue
        self.ack_queue = collections.deque()

        rospy.loginfo(f"[{self.node_id}] Online. Waiting for drone fleet to deploy...")
        self._log_to_file(f"[{self.node_id}] Online. Waiting for drone fleet to deploy...")

        # Wait for drones to spawn (approx 15-20s for 18 drones)
        rospy.sleep(20.0)

        rospy.loginfo(f"[{self.node_id}] Fleet Deployed. Starting 3-Way Handshake (Period: 10s)...")
        self._log_to_file(f"[{self.node_id}] Fleet Deployed. Starting 3-Way Handshake (Period: 10s)...")

        # Periodic "SYN" Broadcast (Step 1)
        self.timer = rospy.Timer(rospy.Duration(10.0), self.broadcast_hello)

        # Async ACK Processor (10Hz)
        self.process_timer = rospy.Timer(rospy.Duration(0.1), self.process_ack_queue)

    def on_mission_complete(self, msg: Bool):
        """Stop chatter when missions are done."""
        if msg.data:
            rospy.loginfo(f"[{self.node_id}] Mission complete signal received. Shutting down beacon.")
            self._log_to_file(f"[{self.node_id}] Mission complete signal received. Shutting down beacon.")
            try:
                self.timer.shutdown()
                self.process_timer.shutdown()
            except Exception:
                pass
            rospy.signal_shutdown("Mission complete")

    def broadcast_hello(self, event):
        """Step 1: Broadcast HELLO (SYN) to all units"""
        msg = "HELLO"
        rospy.loginfo(f"[{self.node_id}] >>> Broadcasting: {msg}")
        self._log_to_file(f"[{self.node_id}] >>> Broadcasting: {msg}")
        self.pub.publish(msg)

    def process_ack_queue(self, event):
        """Process one pending ACK per tick"""
        if self.ack_queue:
            sender_id = self.ack_queue.popleft()
            
            # Step 3: Send ACK (Async)
            ack_msg = f"TOWER_ACK_{sender_id}"
            self.pub.publish(ack_msg)
            rospy.loginfo(f"[{self.node_id}] Received HI from {sender_id}. ACK sent.")
            self._log_to_file(f"[{self.node_id}] Received HI from {sender_id}. ACK sent.")

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
                self._log_to_file(f"[{self.node_id}] HI received from {sender_id} (queued for ACK)")
                
            except IndexError:
                pass

    def _init_log_file(self):
        """Prepare connection log file inside package logs/connection_report.log"""
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        logs_dir = os.path.join(base_dir, 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        return os.path.join(logs_dir, 'connection_report.log')

    def _log_to_file(self, message: str):
        """Append a timestamped message to the connection log file."""
        timestamp = datetime.now().isoformat()
        try:
            with open(self.log_file, 'a') as fh:
                fh.write(f"[{timestamp}] {message}\n")
        except Exception:
            # Logging to file shouldn't break node operation
            pass

if __name__ == '__main__':
    try:
        CentralAgent()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
