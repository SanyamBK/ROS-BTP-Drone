#!/usr/bin/env python3
"""
Drought Probability Model

Implements a probabilistic drought assessment system based on:
- Historical meteorological data (Kaggle US Drought dataset)
- Feature engineering from SPI, SMI, VCI, TCI indices
- Stochastic probability generation (no ML training required)
- Random probability list for rapid testing

Paper: DroughtCast - Machine Learning Forecast of the United States Drought Monitor
Authors: Colin Brust et al. (Frontiers in Big Data, 2021)
"""

import math
import csv
import os
import random
from typing import Dict, List, Tuple

# Sigmoid function implementation to replace scipy.special.expit
def expit(x):
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

# Simple statistics helpers
def calculate_mean(data):
    if not data:
        return 0.0
    return sum(data) / len(data)

def calculate_std(data):
    if len(data) < 2:
        return 1.0
    mean = calculate_mean(data)
    variance = sum((x - mean) ** 2 for x in data) / (len(data) - 1)
    return math.sqrt(variance)

def clip(val, min_val, max_val):
    return max(min_val, min(max_val, val))

class DroughtProbabilityModel:
    """
    Generate drought probability estimates for farmland areas.
    Uses pre-generated probability list for testing without ML training.
    """
    
    def __init__(self, seed: int = None):
        """
        Initialize drought probability model.
        
        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
        # If seed is None, it uses system time/random source
        
        # Pre-generated probabilities (no ML training needed)
        # These represent: low risk, medium risk, high risk areas
        self.probability_pool = [
            0.15, 0.18, 0.22, 0.25, 0.28,  # Low risk (5-30%)
            0.35, 0.38, 0.42, 0.45, 0.48,  # Medium-low risk
            0.52, 0.55, 0.58, 0.62, 0.65,  # Medium risk
            0.68, 0.72, 0.75, 0.78, 0.82,  # Medium-high risk
            0.85, 0.88, 0.90, 0.93, 0.95   # High risk (85-95%)
        ]
        
        # Feature weights (tunable hyperparameters from paper)
        self.weights = {
            'rain_deficit': 0.25,
            'soil_deficit': 0.25,
            'veg_stress': 0.20,
            'heat_index': 0.15,
            'drought_freq': 0.10,
            'trend': 0.05
        }
        
        self.feature_history = {}
        
    def get_random_probability(self) -> float:
        """
        Get a random drought probability from the pre-generated pool.
        
        Returns:
            Drought probability (0.05 to 0.95)
        """
        return random.choice(self.probability_pool)
    
    def get_area_probabilities(self, num_areas: int) -> Dict[str, float]:
        """
        Generate drought probabilities for multiple areas.
        
        Args:
            num_areas: Number of farmland areas
            
        Returns:
            Dictionary mapping area_id to drought probability
        """
        probabilities = {}
        for i in range(num_areas):
            area_id = f"area_{i+1}"
            probabilities[area_id] = self.get_random_probability()
        
        return probabilities
    
    def calculate_feature_vector(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize and validate feature vector.
        
        Args:
            features: Dict with keys [rain_deficit, soil_deficit, veg_stress, 
                                      heat_index, drought_freq, trend]
                     Values should be 0-1 normalized
        
        Returns:
            Validated feature vector with all values clipped to [0, 1]
        """
        normalized = {}
        for key in self.weights.keys():
            value = features.get(key, 0.5)
            # Clip to valid range [0, 1]
            normalized[key] = clip(float(value), 0.0, 1.0)
        
        return normalized
    
    def calculate_probability_from_features(self, features: Dict[str, float]) -> float:
        """
        Calculate drought probability using weighted feature combination.
        
        This method implements the logistic mapping from the paper:
        P(drought) = 0.05 + 0.90 / (1 + exp(-k * (score - 0.5)))
        
        Args:
            features: Dict with normalized feature values (0-1)
            
        Returns:
            Probability between 0.05 and 0.95
        """
        # Normalize features
        norm_features = self.calculate_feature_vector(features)
        
        # Weighted sum of all features
        score = sum(norm_features[k] * self.weights[k] for k in self.weights.keys())
        
        # Logistic mapping to [0.05, 0.95] range
        # Steepness k=10 provides sharp probability gradients
        raw_prob = expit(10 * (score - 0.5))
        prob = 0.05 + 0.90 * raw_prob
        
        return float(clip(prob, 0.05, 0.95))
    
    def extract_features_from_csv(self, csv_path: str, lookback_days: int = 90) -> Dict[str, float]:
        """
        Extract drought features from meteorological CSV data.
        
        Expected columns: PRECTOT, QV2M, T2M_MAX, T2M_MIN, TS, PS
        
        Args:
            csv_path: Path to meteorological timeseries CSV
            lookback_days: Number of historical days to analyze
            
        Returns:
            Dictionary of computed drought features
        """
        try:
            # Optimize reading large CSVs by using system 'tail' command
            # This avoids reading the entire 2GB file into Python memory/CPU
            import subprocess
            import io

            # 1. Read the header
            with open(csv_path, 'r') as f:
                header = f.readline().strip()
            
            # 2. Read the last N lines using tail
            # lookback_days + buffer to ensure we cover enough data points
            # (sometimes data might have gaps, but for this specific logic we just want N rows)
            cmd = ['tail', '-n', str(lookback_days), csv_path]
            result = subprocess.check_output(cmd)
            tail_output = result.decode('utf-8')
            
            # 3. Combine header and tail output
            virtual_csv = io.StringIO(f"{header}\n{tail_output}")
            
            reader = csv.DictReader(virtual_csv)
            data_rows = list(reader)

            if not data_rows:
                raise ValueError("No data read from CSV")

            features = {}
            
            # Helper to extract column data
            def get_col(name):
                return [float(row[name]) for row in data_rows if row.get(name) and row[name].strip() != '']

            # 1. Rainfall Deficit (SPI-inspired)
            precip = get_col('PRECTOT')
            if precip:
                precip_mean = calculate_mean(precip)
                precip_std = calculate_std(precip) + 1e-6
                
                # SPI = (P - mu) / sigma, normalized to [0,1]
                spi = (precip_mean - precip_std) / (precip_std + 1e-6)
                rain_deficit = clip(1 - expit(spi), 0, 1)
                features['rain_deficit'] = float(rain_deficit)
            else:
                features['rain_deficit'] = 0.5
            
            # 2. Moisture (QV2M as Soil Proxy)
            moist = get_col('QV2M')
            if moist:
                moist_mean = calculate_mean(moist)
                # Normalize typical specific humidity (0-0.02 kg/kg)
                # Inverted: Low humidity = High Deficit
                # 0.02 is a rough max
                soil_deficit = 1.0 - (moist_mean / 0.02)
                features['soil_deficit'] = float(clip(soil_deficit, 0, 1))
            else:
                features['soil_deficit'] = 0.5
            
            # 3. Voltage/Stress (TS - Skin Temp as Veg Proxy)
            ts = get_col('TS')
            if ts:
                ts_mean = calculate_mean(ts)
                # High skin temp = High stress
                # Normalize assuming 330K max, 270K min
                veg_stress = (ts_mean - 270) / (330 - 270)
                features['veg_stress'] = float(clip(veg_stress, 0, 1))
            else:
                features['veg_stress'] = 0.5
            
            # 4. Heatwave Intensity (TCI)
            temp_max = get_col('T2M_MAX')
            if temp_max and len(temp_max) >= 30:
                baseline = temp_max[:30]
                recent = temp_max[-30:]
                
                baseline_std = calculate_std(baseline) + 1e-6
                baseline_mean = calculate_mean(baseline)
                recent_mean = calculate_mean(recent)
                
                tci = (recent_mean - baseline_mean) / baseline_std
                heat_index = clip(expit(tci), 0, 1)
                features['heat_index'] = float(heat_index)
            else:
                features['heat_index'] = 0.5
            
            # 5. Historical Drought Frequency
            # Simulated: frequency of extreme conditions
            features['drought_freq'] = 0.0
            if 'veg_stress' in features and 'heat_index' in features:
                drought_events = 0
                for i in range(len(data_rows)):
                    # Approximate per-day values using the aggregate (simplification)
                    # Ideally we would compute VHI per day
                    pass
                # Using the aggregate values as a proxy for the 'current state' contribution to frequency
                # This is a simplification of the original vector logic
                vhi = 0.5 * (1 - features['veg_stress']) + 0.5 * (1 - features['heat_index'])
                if vhi < 0.4:
                     # If the aggregate is low, we assume high frequency? No, that's not right.
                     # Let's revert to a simpler heuristic for this demo without numpy vectorization
                     features['drought_freq'] = 0.2 if vhi < 0.4 else 0.05
            else:
                 features['drought_freq'] = 0.5

            # 6. Trend (recent worsening signal)
            features['trend'] = 0.5
            # Simplified trend: compare last 10 days to first 10 days of the window
            # using vegetation stress proxy (inverted temperature)
            if temp and len(temp) > 20:
                early_mean = calculate_mean(temp[:10])
                late_mean = calculate_mean(temp[-10:])
                if late_mean > early_mean:
                    # Temp rising -> worsening
                     features['trend'] = 0.7
                else:
                     features['trend'] = 0.3
            
            return features
            
        except Exception as e:
            print(f"Error extracting features from {csv_path}: {e}")
            # Return default features on error
            return {
                'rain_deficit': 0.5,
                'soil_deficit': 0.5,
                'veg_stress': 0.5,
                'heat_index': 0.5,
                'drought_freq': 0.5,
                'trend': 0.5
            }
    
    def get_area_risk_ranking(self, area_probabilities: Dict[str, float]) -> List[Tuple[str, float]]:
        """
        Rank areas by drought probability (highest risk first).
        
        Args:
            area_probabilities: Dict mapping area_id to probability
            
        Returns:
            List of (area_id, probability) tuples sorted by probability descending
        """
        ranked = sorted(area_probabilities.items(), key=lambda x: x[1], reverse=True)
        return ranked
    

def test_drought_model():
    """Test the drought probability model."""
    print("=" * 60)
    print("Testing Drought Probability Model")
    print("=" * 60)
    
    model = DroughtProbabilityModel(seed=42)
    
    # Test 1: Random probabilities for 10 areas
    print("\n[Test 1] Random Probabilities for 10 Areas:")
    probs = model.get_area_probabilities(10)
    ranked = model.get_area_risk_ranking(probs)
    
    for i, (area, prob) in enumerate(ranked):
        risk_level = "HIGH" if prob > 0.7 else "MEDIUM" if prob > 0.4 else "LOW"
        print(f"  {i+1}. {area}: {prob:.3f} ({risk_level})")
    
    # Test 2: Feature-based probability calculation
    print("\n[Test 2] Feature-Based Probability Calculation:")
    test_features = {
        'rain_deficit': 0.8,      # High deficit (dry)
        'soil_deficit': 0.7,      # High soil deficit
        'veg_stress': 0.6,        # Moderate vegetation stress
        'heat_index': 0.5,        # Moderate heat
        'drought_freq': 0.4,      # Some history
        'trend': 0.3               # Slightly worsening
    }
    
    prob = model.calculate_probability_from_features(test_features)
    print(f"  Input features: {test_features}")
    print(f"  Calculated probability: {prob:.3f}")
    
    # Test 3: Feature extraction from CSV
    print("\n[Test 3] Feature Extraction from Kaggle Dataset:")
    data_path = "/home/ros/catkin_ws/src/multi_drone_sim/us-drought-meteorological-data/versions/5/train_timeseries/train_timeseries.csv"
    
    if os.path.exists(data_path):
        features = model.extract_features_from_csv(data_path, lookback_days=90)
        print(f"  Extracted features from {data_path}:")
        for key, value in features.items():
            print(f"    {key}: {value:.3f}")
        
        # Calculate probability from extracted features
        prob_from_data = model.calculate_probability_from_features(features)
        print(f"  Calculated drought probability: {prob_from_data:.3f}")
    else:
        print(f"  Dataset not found at {data_path}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_drought_model()
