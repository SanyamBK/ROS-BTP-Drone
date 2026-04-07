#!/usr/bin/env python3

import rospy
import os
from datetime import datetime
from std_msgs.msg import String, Bool, Float32

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

        # Mission end hooks
        self.mission_complete = False
        self.mission_sub = rospy.Subscriber('/mission_complete', Bool, self.on_mission_complete)
        self.coverage_sub = rospy.Subscriber('/system/coverage_pct', Float32, self.on_coverage_update)

        # Heartbeat tracking and Control Channel
        self.last_seen_time = {}
        self.cmd_pub = rospy.Publisher('/central/commands', String, queue_size=10)

        # Retry-based response tracking (replaces async queue)
        # known_drones: all drone IDs that have ever checked in
        # pending_responses: drones we're waiting to hear from in the current HELLO round
        self.known_drones = set()
        self.pending_responses = set()

        rospy.loginfo(f"[{self.node_id}] Online. Waiting for drone fleet to deploy...")
        self._log_to_file(f"[{self.node_id}] Online. Waiting for drone fleet to deploy...")

        # Wait for drones to spawn
        rospy.sleep(20.0)

        rospy.loginfo(f"[{self.node_id}] Fleet Deployed. Starting 3-Way Handshake (Period: 10s)...")
        self._log_to_file(f"[{self.node_id}] Fleet Deployed. Starting 3-Way Handshake (Period: 10s)...")

        # Periodic "SYN" Broadcast (Step 1)
        self.timer = rospy.Timer(rospy.Duration(10.0), self.broadcast_hello)

    def on_mission_complete(self, msg: Bool):
        """Stop chatter when missions are done."""
        if msg.data and not self.mission_complete:
            self.mission_complete = True
            rospy.loginfo(f"[{self.node_id}] Mission complete signal received. Shutting down beacon.")
            self._log_to_file(f"[{self.node_id}] Mission complete signal received. Shutting down beacon.")
            self._shutdown_timer()
            rospy.signal_shutdown("Mission complete")

    def on_coverage_update(self, msg: Float32):
        """Shut down HELLO broadcasts the moment 100% coverage is confirmed."""
        if msg.data >= 100.0 and not self.mission_complete:
            self.mission_complete = True
            rospy.loginfo(f"[{self.node_id}] 100% coverage confirmed via /system/coverage_pct — stopping beacon.")
            self._log_to_file(f"[{self.node_id}] 100% coverage reached. HELLO broadcasts halted.")
            self._shutdown_timer()

    def _shutdown_timer(self):
        """Safely stop the periodic HELLO timer."""
        try:
            self.timer.shutdown()
        except Exception:
            pass

    def broadcast_hello(self, event):
        """Step 1: Broadcast HELLO (SYN) to all units"""
        now = rospy.Time.now()
        dead_agents = []

        # Check for missing heartbeats before broadcasting
        for sender_id, t in self.last_seen_time.copy().items():
            if (now - t).to_sec() > 15.0:
                dead_agents.append(sender_id)
                del self.last_seen_time[sender_id]
                self.known_drones.discard(sender_id)

        for dead_id in dead_agents:
            rospy.logerr(f"[{self.node_id}] TIMEOUT: {dead_id} missed heartbeat! Requesting Reserve!")
            self._log_to_file(f"[{self.node_id}] TIMEOUT: {dead_id} missed heartbeat. Requesting Reserve!")
            self.cmd_pub.publish("DEPLOY_RESERVE")

        # Set who we expect to hear back from this round
        self.pending_responses = self.known_drones.copy()

        msg = "HELLO"
        rospy.loginfo(f"[{self.node_id}] >>> Broadcasting: {msg} ({len(self.known_drones)} known agents, {len(self.pending_responses)} awaiting response)")
        self._log_to_file(f"[{self.node_id}] >>> Broadcasting: {msg}")
        self.pub.publish(msg)

        # Schedule retry check in 2.2s
        rospy.Timer(rospy.Duration(2.2), self.check_and_retry, oneshot=True)

    def check_and_retry(self, event):
        """2.2s after HELLO: send targeted retries to any drone that hasn't responded yet"""
        if not self.pending_responses:
            return
        for sender_id in list(self.pending_responses):
            rospy.logwarn(f"[{self.node_id}] No response from {sender_id} after 2.2s. Retrying...")
            self._log_to_file(f"[{self.node_id}] Retry HELLO sent to {sender_id}")
            self.pub.publish(f"HELLO_RETRY_{sender_id}")

    def callback(self, msg):
        """Handle responses from Agents"""
        data = msg.data

        # Step 2 Receive: Listen for HI (SYN-ACK)
        if "AGENT_HI" in data:
            try:
                # Format: AGENT_HI_<ID>
                sender_id = data.split("_HI_")[1]

                # Update heartbeat timestamp
                self.last_seen_time[sender_id] = rospy.Time.now()

                # Register as known
                self.known_drones.add(sender_id)

                # Remove from pending — they responded this round
                self.pending_responses.discard(sender_id)

                # ACK immediately — no queue
                ack_msg = f"TOWER_ACK_{sender_id}"
                self.pub.publish(ack_msg)
                rospy.loginfo(f"[{self.node_id}] HI from {sender_id} — ACK sent immediately.")
                self._log_to_file(f"[{self.node_id}] HI from {sender_id} — ACK sent immediately.")

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
