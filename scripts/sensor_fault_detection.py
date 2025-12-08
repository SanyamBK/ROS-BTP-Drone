#!/usr/bin/env python3
"""
Sensor Fault Detection & Multi-Drone Verification

Implements:
- Sensor fault detection (comparing model prediction vs sensor reading)
- Probability merging when multiple drones verify the same area
- Auditor drone deployment for validation
- Inverse-variance weighting for sensor fusion

Paper: DroughtCast - Machine Learning Forecast of the United States Drought Monitor
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class SensorStatus(Enum):
    """Sensor operational status."""
    HEALTHY = "healthy"
    SUSPICIOUS = "suspicious"
    FAULTY = "faulty"
    VALIDATING = "validating"
    CORRECTED = "corrected"


@dataclass
class SensorReading:
    """Single sensor measurement."""
    drone_id: int
    area_id: str
    probability: float
    noise_std: float
    timestamp: float
    status: SensorStatus = SensorStatus.HEALTHY
    is_auditor: bool = False


@dataclass
class AreaMeasurement:
    """Aggregated measurement for an area."""
    area_id: str
    model_probability: float
    sensor_readings: List[SensorReading]
    fused_probability: Optional[float] = None
    confidence: float = 0.0


class SensorFaultDetector:
    """
    Detect faulty sensors by comparing model predictions with drone measurements.
    """
    
    def __init__(self, fault_threshold: float = 0.15, 
                 sensitivity: float = 2.0):
        """
        Initialize fault detector.
        
        Args:
            fault_threshold: Maximum acceptable deviation (0.15 = 15%)
            sensitivity: Sensitivity multiplier for fault detection (1.5-2.5)
        """
        self.fault_threshold = fault_threshold
        self.sensitivity = sensitivity
        self.detection_history = {}
    
    def detect_fault(self, model_prob: float, sensor_prob: float, 
                    sensor_noise: float) -> Tuple[bool, str, float]:
        """
        Detect if sensor reading is faulty.
        
        Uses statistical hypothesis testing:
        If |sensor - model| > 2sigma + threshold, mark as faulty
        
        Args:
            model_prob: Drought probability from ML model
            sensor_prob: Probability measured by drone sensor
            sensor_noise: Known sensor noise standard deviation
            
        Returns:
            Tuple of (is_faulty, reason, confidence_score)
            - is_faulty: True if fault detected
            - reason: Human-readable explanation
            - confidence_score: 0-1 confidence in fault detection
        """
        error = abs(sensor_prob - model_prob)
        
        # Expected error is 2sigma for 95% confidence interval
        expected_error = 2 * sensor_noise
        
        # Fault threshold accounting for sensitivity
        adaptive_threshold = (expected_error + self.fault_threshold) * self.sensitivity
        
        # Calculate confidence in fault detection
        if error > adaptive_threshold:
            # High confidence fault
            confidence = min(1.0, error / (adaptive_threshold * 1.5))
            reason = (f"Deviation {error:.3f} exceeds threshold "
                     f"{adaptive_threshold:.3f} (expected: {expected_error:.3f})")
            return (True, reason, confidence)
        
        elif error > expected_error + self.fault_threshold * 0.5:
            # Suspicious reading - low confidence fault
            confidence = min(1.0, (error - expected_error) / (self.fault_threshold))
            reason = (f"Suspicious deviation {error:.3f} (expected: {expected_error:.3f})")
            return (False, reason, confidence * 0.5)
        
        else:
            # Reading is within acceptable range
            return (False, "Reading within acceptable range", 0.0)
    
    def log_detection(self, drone_id: int, area_id: str, is_faulty: bool, 
                     confidence: float):
        """
        Log fault detection event for analysis.
        
        Args:
            drone_id: ID of drone
            area_id: ID of area measured
            is_faulty: Whether fault was detected
            confidence: Confidence score
        """
        key = f"drone_{drone_id}_area_{area_id}"
        if key not in self.detection_history:
            self.detection_history[key] = []
        
        self.detection_history[key].append({
            'faulty': is_faulty,
            'confidence': confidence
        })
    
    def get_sensor_reliability(self, drone_id: int) -> float:
        """
        Calculate overall reliability of a drone's sensor.
        
        Args:
            drone_id: ID of drone
            
        Returns:
            Reliability score (0-1, where 1 = perfect sensor)
        """
        drone_readings = [
            v for k, v in self.detection_history.items()
            if k.startswith(f"drone_{drone_id}_")
        ]
        
        if not drone_readings:
            return 1.0  # Assume healthy if no history
        
        all_readings = []
        for readings_list in drone_readings:
            all_readings.extend(readings_list)
        
        if not all_readings:
            return 1.0
        
        fault_rate = sum(1 for r in all_readings if r['faulty']) / len(all_readings)
        reliability = 1.0 - fault_rate
        
        return float(np.clip(reliability, 0.0, 1.0))


class SensorFusion:
    """
    Merge multiple sensor readings using inverse-variance weighting.
    """
    
    @staticmethod
    def inverse_variance_fusion(readings: List[SensorReading]) -> Tuple[float, float]:
        """
        Fuse multiple sensor readings using inverse-variance weighting.
        
        Optimal sensor fusion formula:
        - Weight_i = 1 / sigma_i2
        - Fused_probability = SUM(Weight_i * P_i) / SUM(Weight_i)
        - Fused_std = 1 / sqrt(SUM(Weight_i))
        
        Args:
            readings: List of SensorReading objects
            
        Returns:
            Tuple of (fused_probability, fused_noise_std)
        """
        if not readings:
            return 0.5, 1.0
        
        if len(readings) == 1:
            return readings[0].probability, readings[0].noise_std
        
        # Calculate inverse-variance weights
        weights = []
        weighted_sum = 0.0
        weight_sum = 0.0
        
        for reading in readings:
            noise_var = reading.noise_std ** 2 + 1e-6  # Avoid division by zero
            weight = 1.0 / noise_var
            weights.append(weight)
            weighted_sum += weight * reading.probability
            weight_sum += weight
        
        # Compute fused estimates
        fused_prob = weighted_sum / weight_sum
        fused_std = 1.0 / np.sqrt(weight_sum + 1e-6)
        
        return float(np.clip(fused_prob, 0.05, 0.95)), float(fused_std)
    
    @staticmethod
    def bayesian_fusion(readings: List[SensorReading], 
                       prior_prob: float = 0.5) -> float:
        """
        Fuse readings using Bayesian inference.
        
        Updates prior belief based on observed readings.
        
        Args:
            readings: List of sensor readings
            prior_prob: Prior belief about drought probability
            
        Returns:
            Posterior probability estimate
        """
        if not readings:
            return prior_prob
        
        posterior = prior_prob
        
        for reading in readings:
            # Likelihood of observing this reading given drought probability
            likelihood = np.exp(-0.5 * ((reading.probability - posterior) / reading.noise_std) ** 2)
            
            # Update posterior (simplified Bayesian update)
            posterior = (likelihood * reading.probability + (1 - likelihood) * posterior) / 2
        
        return float(np.clip(posterior, 0.05, 0.95))
    
    @staticmethod
    def weighted_average_fusion(readings: List[SensorReading], 
                               reliability_scores: Dict[int, float]) -> float:
        """
        Fuse readings using drone reliability scores as weights.
        
        Drones with better track record (higher reliability) contribute more.
        
        Args:
            readings: List of sensor readings
            reliability_scores: Dict mapping drone_id to reliability (0-1)
            
        Returns:
            Weighted average probability
        """
        if not readings:
            return 0.5
        
        weighted_sum = 0.0
        reliability_sum = 0.0
        
        for reading in readings:
            reliability = reliability_scores.get(reading.drone_id, 0.5)
            weighted_sum += reliability * reading.probability
            reliability_sum += reliability
        
        if reliability_sum == 0:
            return np.mean([r.probability for r in readings])
        
        return float(np.clip(weighted_sum / reliability_sum, 0.05, 0.95))


class DroneVerificationSystem:
    """
    Manage multi-drone verification of measurements and sensor validation.
    """
    
    def __init__(self):
        """Initialize verification system."""
        self.area_measurements: Dict[str, AreaMeasurement] = {}
        self.fault_detector = SensorFaultDetector()
        self.sensor_fusion = SensorFusion()
        self.auditor_deployments: Dict[str, List[int]] = {}  # area_id -> auditor_ids
    
    def add_measurement(self, reading: SensorReading, model_probability: float):
        """
        Add a drone sensor measurement for an area.
        
        Args:
            reading: SensorReading object
            model_probability: Expected probability from ML model
        """
        area_id = reading.area_id
        
        # Initialize area measurement if needed
        if area_id not in self.area_measurements:
            self.area_measurements[area_id] = AreaMeasurement(
                area_id=area_id,
                model_probability=model_probability,
                sensor_readings=[]
            )
        
        measurement = self.area_measurements[area_id]
        measurement.sensor_readings.append(reading)
        measurement.model_probability = model_probability
        
        # Perform fault detection
        is_faulty, reason, confidence = self.fault_detector.detect_fault(
            model_probability,
            reading.probability,
            reading.noise_std
        )
        
        if is_faulty:
            reading.status = SensorStatus.FAULTY
            print(f"[FAULT DETECTED] Drone {reading.drone_id} at {area_id}: {reason}")
            print(f"  Confidence: {confidence:.3f}")
            self._trigger_auditor_deployment(reading, measurement)
        else:
            reading.status = SensorStatus.HEALTHY
        
        # Log detection
        self.fault_detector.log_detection(reading.drone_id, area_id, is_faulty, confidence)
    
    def _trigger_auditor_deployment(self, faulty_reading: SensorReading, 
                                    measurement: AreaMeasurement):
        """
        Trigger deployment of auditor drone to validate faulty sensor.
        
        Args:
            faulty_reading: The faulty sensor reading
            measurement: The area measurement
        """
        area_id = measurement.area_id
        
        if area_id not in self.auditor_deployments:
            self.auditor_deployments[area_id] = []
        
        print(f"[AUDITOR DEPLOYMENT] Area {area_id} needs validation")
        print(f"  Faulty drone: {faulty_reading.drone_id}")
        print(f"  Faulty reading: {faulty_reading.probability:.3f}")
        print(f"  Model prediction: {measurement.model_probability:.3f}")
    
    def merge_auditor_verification(self, area_id: str, original_drone_id: int, 
                                   auditor_id: int, auditor_reading: float,
                                   auditor_noise: float) -> float:
        """
        Merge auditor verification with original faulty reading.
        
        Uses inverse-variance weighting to combine measurements.
        
        Args:
            area_id: Area being verified
            original_drone_id: ID of original drone with faulty sensor
            auditor_id: ID of auditor drone
            auditor_reading: Probability measured by auditor
            auditor_noise: Noise std of auditor sensor
            
        Returns:
            Fused probability estimate
        """
        if area_id not in self.area_measurements:
            print(f"[ERROR] Area {area_id} not found in measurements")
            return 0.5
        
        measurement = self.area_measurements[area_id]
        
        # Find original faulty reading
        original_reading = None
        for reading in measurement.sensor_readings:
            if reading.drone_id == original_drone_id and not reading.is_auditor:
                original_reading = reading
                break
        
        if not original_reading:
            print(f"[ERROR] Could not find original reading from drone {original_drone_id}")
            return auditor_reading
        
        # Create auditor reading object
        auditor_reading_obj = SensorReading(
            drone_id=auditor_id,
            area_id=area_id,
            probability=auditor_reading,
            noise_std=auditor_noise,
            timestamp=0.0,  # Updated by caller
            status=SensorStatus.VALIDATING,
            is_auditor=True
        )
        
        # Perform inverse-variance fusion
        readings_to_fuse = [original_reading, auditor_reading_obj]
        fused_prob, fused_std = self.sensor_fusion.inverse_variance_fusion(readings_to_fuse)
        
        # Update measurement
        measurement.fused_probability = fused_prob
        measurement.confidence = 1.0 - fused_std  # Higher precision = higher confidence
        
        # Mark original reading as corrected
        original_reading.status = SensorStatus.CORRECTED
        auditor_reading_obj.status = SensorStatus.CORRECTED
        measurement.sensor_readings.append(auditor_reading_obj)
        
        print(f"[SENSOR FUSION] Area {area_id} verification complete")
        print(f"  Original (faulty) reading: {original_reading.probability:.3f} (sigma={original_reading.noise_std:.3f})")
        print(f"  Auditor reading: {auditor_reading:.3f} (sigma={auditor_noise:.3f})")
        print(f"  Fused probability: {fused_prob:.3f}")
        print(f"  Fused std dev: {fused_std:.3f}")
        print(f"  Confidence: {measurement.confidence:.3f}")
        
        return fused_prob
    
    def get_area_probability(self, area_id: str, use_fusion: bool = True) -> float:
        """
        Get final probability estimate for an area.
        
        Args:
            area_id: Area ID
            use_fusion: Use fused probability if available
            
        Returns:
            Probability estimate (0.05-0.95)
        """
        if area_id not in self.area_measurements:
            return 0.5
        
        measurement = self.area_measurements[area_id]
        
        # Return fused probability if available and requested
        if use_fusion and measurement.fused_probability is not None:
            return measurement.fused_probability
        
        # Otherwise, fuse all available readings
        if measurement.sensor_readings:
            fused_prob, _ = self.sensor_fusion.inverse_variance_fusion(
                measurement.sensor_readings
            )
            return fused_prob
        
        # Fallback to model prediction
        return measurement.model_probability
    
    def get_measurement_summary(self, area_id: str) -> Dict:
        """
        Get summary of all measurements for an area.
        
        Args:
            area_id: Area ID
            
        Returns:
            Dictionary with measurement summary
        """
        if area_id not in self.area_measurements:
            return {}
        
        measurement = self.area_measurements[area_id]
        
        return {
            'area_id': area_id,
            'model_probability': measurement.model_probability,
            'num_readings': len(measurement.sensor_readings),
            'num_faults': sum(1 for r in measurement.sensor_readings 
                            if r.status == SensorStatus.FAULTY),
            'fused_probability': measurement.fused_probability,
            'confidence': measurement.confidence,
            'readings': [
                {
                    'drone_id': r.drone_id,
                    'probability': r.probability,
                    'status': r.status.value,
                    'is_auditor': r.is_auditor
                }
                for r in measurement.sensor_readings
            ]
        }


def test_fault_detection():
    """Test sensor fault detection system."""
    print("=" * 60)
    print("Testing Sensor Fault Detection & Verification System")
    print("=" * 60)
    
    detector = SensorFaultDetector(fault_threshold=0.15, sensitivity=2.0)
    verifier = DroneVerificationSystem()
    
    # Test 1: Detect faulty sensor
    print("\n[Test 1] Detect Faulty Sensor Reading:")
    model_prob = 0.60
    sensor_prob_good = 0.58
    sensor_prob_bad = 0.25
    noise = 0.05
    
    is_faulty, reason, conf = detector.detect_fault(model_prob, sensor_prob_good, noise)
    print(f"  Good reading ({sensor_prob_good:.3f}): Faulty={is_faulty}, Confidence={conf:.3f}")
    
    is_faulty, reason, conf = detector.detect_fault(model_prob, sensor_prob_bad, noise)
    print(f"  Bad reading ({sensor_prob_bad:.3f}): Faulty={is_faulty}, Confidence={conf:.3f}")
    print(f"    Reason: {reason}")
    
    # Test 2: Inverse-variance fusion
    print("\n[Test 2] Inverse-Variance Sensor Fusion:")
    readings = [
        SensorReading(drone_id=1, area_id="area_1", probability=0.55, noise_std=0.05, timestamp=0.0),
        SensorReading(drone_id=2, area_id="area_1", probability=0.62, noise_std=0.03, timestamp=1.0),
        SensorReading(drone_id=3, area_id="area_1", probability=0.58, noise_std=0.04, timestamp=2.0),
    ]
    
    fused_prob, fused_std = SensorFusion.inverse_variance_fusion(readings)
    print(f"  Individual readings:")
    for r in readings:
        print(f"    Drone {r.drone_id}: {r.probability:.3f} (sigma={r.noise_std:.3f})")
    print(f"  Fused probability: {fused_prob:.3f}")
    print(f"  Fused std dev: {fused_std:.3f}")
    
    # Test 3: Multi-drone verification with auditor
    print("\n[Test 3] Multi-Drone Verification with Auditor:")
    model_prob = 0.65
    
    # Add initial reading (slightly off)
    initial_reading = SensorReading(
        drone_id=1, area_id="area_2", probability=0.35,  # Faulty reading
        noise_std=0.08, timestamp=0.0
    )
    verifier.add_measurement(initial_reading, model_prob)
    
    # Merge auditor verification
    fused = verifier.merge_auditor_verification(
        area_id="area_2",
        original_drone_id=1,
        auditor_id=10,
        auditor_reading=0.67,
        auditor_noise=0.03
    )
    
    summary = verifier.get_measurement_summary("area_2")
    print(f"  Measurement summary:")
    print(f"    Model probability: {summary['model_probability']:.3f}")
    print(f"    Fused probability: {summary['fused_probability']:.3f}")
    print(f"    Confidence: {summary['confidence']:.3f}")
    print(f"    Readings: {summary['num_readings']} (Faults: {summary['num_faults']})")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_fault_detection()
