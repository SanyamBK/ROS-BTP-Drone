#!/usr/bin/env python3

import rospy
import threading
import random
import os
from datetime import datetime
from std_msgs.msg import String, Int32

class DroneCommManager:
    def __init__(self):
        rospy.init_node('drone_comm_manager')
        
        # Read num_drones from ROS parameter (set by launch file)
        self.num_drones = rospy.get_param('~num_drones', 10)
        self.drones = []

        # Connection log file (shared with central agent)
        self.log_file = self._init_log_file()
        
        self.pub = rospy.Publisher('/comm/agents', String, queue_size=20)
        self.sub = rospy.Subscriber('/central/comm', String, self.callback)
        self.death_sub = rospy.Subscriber('/swarm/drone_death', Int32, self.death_callback)
        
        rospy.loginfo(f"[DroneNet] Initializing fleet of {self.num_drones} drones...")
        self._log_to_file(f"[DroneNet] Ready with {self.num_drones} drones")
        
        # Create virtual drone states
        for i in range(self.num_drones):
            self.drones.append(VirtualDrone(i, self.pub, self._log_to_file))

    def callback(self, msg):
        """Handle messages from Central Tower"""
        cmd = msg.data
        
        if cmd == "HELLO":
            # Step 1 Receive: Tower initiates handshake
            # Propagate to all drones to respond
            self._log_to_file("[DroneNet] HELLO received from central")
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
                        self._log_to_file(f"[{self.drones[idx].name}] ACK received from central")
            except:
                pass

        elif cmd.startswith("HELLO_RETRY_"):
            # Targeted retry: central agent didn't hear back from this specific drone
            target_name = cmd[len("HELLO_RETRY_"):]
            for drone in self.drones:
                if drone.name == target_name:
                    rospy.logwarn(f"[DroneNet] Retry received for {target_name}. Re-responding...")
                    drone.respond_to_hello()
                    break

    def death_callback(self, msg):
        drone_id = msg.data
        if 0 <= drone_id < self.num_drones:
            self.drones[drone_id].is_dead = True
            self._log_to_file(f"[DroneNet] Hardware failure recorded for DRONE_{drone_id} - Heartbeat silenced.")

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

class VirtualDrone:
    def __init__(self, drone_id, pub, log_func):
        self.id = drone_id
        self.name = f"DRONE_{drone_id}"
        self.pub = pub
        self.log = log_func
        self.connected = False
        self.is_dead = False

    def respond_to_hello(self):
        """Step 2: Send HI (SYN-ACK)"""
        if self.is_dead:
            return
            
        # Add small random delay to prevent network congestion/collision in sim
        delay = random.uniform(0.1, 2.0)
        
        def send():
            rospy.sleep(delay)
            # LOG THE MIDDLE PART
            rospy.loginfo(f"[{self.name}] Heard HELLO. Sending HI...")
            if self.log:
                self.log(f"[{self.name}] Heard HELLO. Sending HI...")
            msg = f"AGENT_HI_{self.name}"
            self.pub.publish(msg)
        
        threading.Thread(target=send).start()

    def receive_ack(self):
        """Step 3 Complete: Connection Established"""
        if not self.connected:
            rospy.loginfo(f"[{self.name}] Connection Established! (ACK Received)")
            if self.log:
                self.log(f"[{self.name}] Connection Established! (ACK Received)")
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
