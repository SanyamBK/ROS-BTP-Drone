#!/usr/bin/env python3
"""
Integrated Drought Monitoring System Demo

Demonstrates the complete workflow:
1. Generate drought probabilities for areas
2. Prioritize areas by risk
3. Allocate drones to areas
4. Simulate sensor readings
5. Detect faults and deploy auditors
6. Merge auditor verifications

Paper: DroughtCast - Machine Learning Forecast of the United States Drought Monitor
"""

import sys
sys.path.insert(0, '/home/ros/catkin_ws/src/multi_drone_sim/scripts')

from drought_probability_model import DroughtProbabilityModel
from sensor_fault_detection import (
    SensorFaultDetector, SensorFusion, DroneVerificationSystem,
    SensorReading, SensorStatus
)
from area_allocation import (
    AreaPrioritizer, DynamicDroneAllocator, Area, Drone, DroneRole
)
import numpy as np
import random


class IntegratedDroughtMonitoringSystem:
    """Complete multi-drone drought monitoring system."""
    
    def __init__(self, num_areas: int = 10, num_drones: int = 18):
        """
        Initialize the integrated system.
        
        Args:
            num_areas: Number of farmland areas
            num_drones: Number of drones in fleet
        """
        self.num_areas = num_areas
        self.num_drones = num_drones
        
        # Initialize components
        self.probability_model = DroughtProbabilityModel(seed=42)
        self.area_prioritizer = AreaPrioritizer()
        self.drone_allocator = DynamicDroneAllocator(
            total_drones=num_drones,
            total_areas=num_areas,
            min_drones_per_area=1,
            max_drones_per_area=3,
            reserve_percentage=0.1
        )
        self.verification_system = DroneVerificationSystem()
        
        # Data structures
        self.areas = []
        self.drones = []
        self.current_allocations = {}
        self.mission_log = []
    
    def initialize_areas(self, seed: int = 42):
        """
        Initialize farmland areas with scattered positions and crops.
        
        Args:
            seed: Random seed
        """
        random.seed(seed)
        np.random.seed(seed)
        
        # Area definitions (from our 10-area system)
        area_definitions = [
            ("area_1", -12.0, 9.0, "Wheat"),
            ("area_2", -2.0, 11.5, "Corn"),
            ("area_3", 11.0, 7.5, "Soybean"),
            ("area_4", -11.0, 1.5, "Wheat"),
            ("area_5", 1.0, -1.0, "Barley"),
            ("area_6", 9.0, -2.0, "Corn"),
            ("area_7", -8.0, -9.5, "Soybean"),
            ("area_8", 0.5, -12.0, "Wheat"),
            ("area_9", 8.5, -11.0, "Corn"),
            ("area_10", 3.5, 4.0, "Barley"),
        ]
        
        # Get probabilities from model
        probabilities = self.probability_model.get_area_probabilities(self.num_areas)
        
        # Create areas
        for i, (area_id, x, y, crop) in enumerate(area_definitions[:self.num_areas]):
            area = Area(
                area_id=area_id,
                drought_probability=probabilities.get(area_id, 0.5),
                size_m2=random.randint(5000, 15000),
                coverage_radius=random.uniform(40, 65),
                x_center=x,
                y_center=y,
                crop_type=crop
            )
            self.areas.append(area)
    
    def initialize_drones(self, seed: int = 42):
        """
        Initialize drone fleet.
        
        Args:
            seed: Random seed
        """
        random.seed(seed)
        
        for i in range(self.num_drones):
            drone = Drone(
                drone_id=i,
                role=DroneRole.IDLE,
                sensor_noise=random.uniform(0.03, 0.08),
                is_operational=True,
                reliability_score=1.0
            )
            self.drones.append(drone)
    
    def run_allocation(self):
        """Allocate drones to areas based on risk."""
        print("\n" + "=" * 70)
        print("PHASE 1: AREA PRIORITIZATION & DRONE ALLOCATION")
        print("=" * 70)
        
        # Rank areas
        ranked_areas = self.area_prioritizer.rank_areas_by_risk(self.areas)
        
        print("\nArea Risk Ranking:")
        for area in ranked_areas[:5]:  # Show top 5
            risk_level = "HIGH" if area.drought_probability > 0.70 else \
                        "MEDIUM" if area.drought_probability > 0.40 else "LOW"
            print(f"  Rank {area.priority_rank + 1}: {area.area_id} "
                  f"({area.crop_type}, {area.drought_probability:.3f}) - {risk_level}")
        
        # Allocate drones
        allocation_result = self.drone_allocator.allocate_drones(self.areas, self.drones)
        self.current_allocations = allocation_result.allocations
        
        print(f"\nAllocation Summary:")
        print(f"  Total drones allocated: {allocation_result.allocation_summary['total_allocated']}")
        print(f"  Drones in reserve: {allocation_result.allocation_summary['total_reserves']}")
        print(f"  Areas covered: {allocation_result.allocation_summary['areas_covered']}/{self.num_areas}")
        
        print(f"\nTop 5 Areas with Drone Assignments:")
        for entry in allocation_result.allocation_summary['allocation_table'][:5]:
            print(f"  {entry['area_id']} (Risk: {entry['probability']:.3f})")
            print(f"    Assigned drones: {entry['drone_ids']}")
        
        return allocation_result
    
    def run_mission_simulation(self, num_measurements: int = 3):
        """
        Simulate drone measurements and fault detection.
        
        Args:
            num_measurements: Number of sensor measurements to simulate
        """
        print("\n" + "=" * 70)
        print("PHASE 2: MISSION EXECUTION & SENSOR MEASUREMENTS")
        print("=" * 70)
        
        faulty_areas = []  # Track areas with detected faults
        
        for area in self.areas[:5]:  # Simulate for first 5 areas
            print(f"\n{area.area_id} (Risk: {area.drought_probability:.3f}, Crop: {area.crop_type}):")
            
            # Get assigned drones for this area
            assigned_drone_ids = self.current_allocations.get(area.area_id, [])
            
            if not assigned_drone_ids:
                print(f"  No drones assigned")
                continue
            
            # Simulate measurements from each drone
            for i, drone_id in enumerate(assigned_drone_ids[:num_measurements]):
                drone = self.drones[drone_id]
                
                # Simulate measurement: mostly accurate, with noise
                # 10% chance of being faulty (bad sensor)
                is_faulty_sensor = random.random() < 0.10
                
                if is_faulty_sensor:
                    # Faulty sensor reads wrong value
                    measured_prob = area.drought_probability + random.uniform(-0.3, -0.1)
                else:
                    # Good sensor with normal noise
                    measured_prob = area.drought_probability + np.random.normal(0, drone.sensor_noise)
                
                measured_prob = np.clip(measured_prob, 0.05, 0.95)
                
                # Create sensor reading
                reading = SensorReading(
                    drone_id=drone_id,
                    area_id=area.area_id,
                    probability=measured_prob,
                    noise_std=drone.sensor_noise,
                    timestamp=0.0,
                    is_auditor=False
                )
                
                # Add to verification system
                self.verification_system.add_measurement(reading, area.drought_probability)
                
                # Log measurement
                status_str = "FAULTY" if is_faulty_sensor else "GOOD"
                print(f"  Drone {drone_id}: {measured_prob:.3f} ({status_str}, σ={drone.sensor_noise:.4f})")
                
                # Track faulty readings
                if is_faulty_sensor:
                    faulty_areas.append((area.area_id, drone_id))
        
        return faulty_areas
    
    def run_auditor_verification(self, faulty_areas: list):
        """
        Deploy auditors to verify faulty sensors.
        
        Args:
            faulty_areas: List of (area_id, drone_id) tuples with faults
        """
        if not faulty_areas:
            print("\n" + "=" * 70)
            print("PHASE 3: AUDITOR VERIFICATION")
            print("=" * 70)
            print("\nNo faulty sensors detected. Skipping auditor deployment.")
            return
        
        print("\n" + "=" * 70)
        print("PHASE 3: AUDITOR VERIFICATION & SENSOR FUSION")
        print("=" * 70)
        
        for area_id, faulty_drone_id in faulty_areas[:2]:  # Verify first 2 faults
            area = next((a for a in self.areas if a.area_id == area_id), None)
            
            if not area:
                continue
            
            print(f"\n{area_id}: Deploying auditor to verify drone {faulty_drone_id}")
            
            # Deploy auditor with lower noise
            auditor_id = 100 + faulty_drone_id  # Use different ID range
            auditor_noise = 0.02  # Lower noise than faulty sensor
            
            # Auditor measures closer to true value
            auditor_prob = area.drought_probability + np.random.normal(0, auditor_noise)
            auditor_prob = np.clip(auditor_prob, 0.05, 0.95)
            
            print(f"  Auditor {auditor_id} reading: {auditor_prob:.3f} (σ={auditor_noise:.4f})")
            
            # Merge auditor verification
            fused_prob = self.verification_system.merge_auditor_verification(
                area_id=area_id,
                original_drone_id=faulty_drone_id,
                auditor_id=auditor_id,
                auditor_reading=auditor_prob,
                auditor_noise=auditor_noise
            )
            
            # Log final result
            summary = self.verification_system.get_measurement_summary(area_id)
            self.mission_log.append({
                'area_id': area_id,
                'faulty_drone': faulty_drone_id,
                'auditor_drone': auditor_id,
                'fused_probability': fused_prob,
                'confidence': summary['confidence']
            })
    
    def generate_mission_report(self):
        """Generate comprehensive mission report."""
        print("\n" + "=" * 70)
        print("MISSION REPORT & SUMMARY")
        print("=" * 70)
        
        print(f"\nSystem Configuration:")
        print(f"  Total drones: {self.num_drones}")
        print(f"  Total areas: {self.num_areas}")
        print(f"  Average drones per area: {self.num_drones / self.num_areas:.1f}")
        
        print(f"\nArea Statistics:")
        probabilities = [a.drought_probability for a in self.areas]
        print(f"  Avg drought probability: {np.mean(probabilities):.3f}")
        print(f"  High-risk areas (>70%): {sum(1 for p in probabilities if p > 0.70)}")
        print(f"  Medium-risk areas (40-70%): {sum(1 for p in probabilities if 0.40 < p <= 0.70)}")
        print(f"  Low-risk areas (<40%): {sum(1 for p in probabilities if p < 0.40)}")
        
        if self.mission_log:
            print(f"\nVerification Results:")
            print(f"  Faults detected and corrected: {len(self.mission_log)}")
            
            avg_confidence = np.mean([m['confidence'] for m in self.mission_log])
            print(f"  Average verification confidence: {avg_confidence:.3f}")
            
            for log in self.mission_log[:3]:
                print(f"\n  {log['area_id']}:")
                print(f"    Faulty drone: {log['faulty_drone']}")
                print(f"    Auditor drone: {log['auditor_drone']}")
                print(f"    Fused probability: {log['fused_probability']:.3f}")
                print(f"    Confidence: {log['confidence']:.3f}")
        
        print("\n" + "=" * 70)


def main():
    """Run complete demo."""
    print("\n" + "=" * 70)
    print("INTEGRATED MULTI-DRONE DROUGHT MONITORING SYSTEM")
    print("Paper: DroughtCast - Machine Learning Forecast of Drought Monitor")
    print("=" * 70)
    
    # Initialize system
    system = IntegratedDroughtMonitoringSystem(num_areas=10, num_drones=18)
    
    # Phase 1: Setup
    print("\nInitializing system...")
    system.initialize_areas()
    system.initialize_drones()
    
    # Phase 2: Allocation
    allocation_result = system.run_allocation()
    
    # Phase 3: Simulation
    faulty_areas = system.run_mission_simulation(num_measurements=3)
    
    # Phase 4: Verification
    system.run_auditor_verification(faulty_areas)
    
    # Phase 5: Report
    system.generate_mission_report()
    
    print("\n✓ Demo complete!\n")


if __name__ == "__main__":
    main()
