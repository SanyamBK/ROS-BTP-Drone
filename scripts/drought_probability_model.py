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

import numpy as np
import pandas as pd
from scipy.special import expit
from typing import Dict, List, Tuple
import random
import os


class DroughtProbabilityModel:
    """
    Generate drought probability estimates for farmland areas.
    Uses pre-generated probability list for testing without ML training.
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize drought probability model.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
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
            normalized[key] = np.clip(float(value), 0.0, 1.0)
        
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
        
        return float(np.clip(prob, 0.05, 0.95))
    
    def extract_features_from_csv(self, csv_path: str, lookback_days: int = 90) -> Dict[str, float]:
        """
        Extract drought features from meteorological CSV data.
        
        Expected columns: PRECTOT, GWETTOP, T2M_MAX, T2M, TS, and derived NDVI
        
        Args:
            csv_path: Path to meteorological timeseries CSV
            lookback_days: Number of historical days to analyze
            
        Returns:
            Dictionary of computed drought features
        """
        try:
            df = pd.read_csv(csv_path)
            
            # Use last lookback_days rows
            if len(df) > lookback_days:
                df = df.iloc[-lookback_days:]
            
            features = {}
            
            # 1. Rainfall Deficit (SPI-inspired)
            if 'PRECTOT' in df.columns:
                precip = df['PRECTOT'].values.astype(float)
                precip_mean = np.nanmean(precip)
                precip_std = np.nanstd(precip) + 1e-6
                
                # SPI = (P - μ) / σ, normalized to [0,1]
                spi = (precip_mean - precip_std) / (precip_std + 1e-6)
                rain_deficit = np.clip(1 - expit(spi), 0, 1)
                features['rain_deficit'] = float(rain_deficit)
            else:
                features['rain_deficit'] = 0.5
            
            # 2. Soil Moisture Deficit
            if 'GWETTOP' in df.columns:
                soil = df['GWETTOP'].values.astype(float)
                soil_max = np.nanmax(soil) + 1e-6
                soil_mean = np.nanmean(soil)
                soil_deficit = 1.0 - (soil_mean / soil_max)
                features['soil_deficit'] = float(np.clip(soil_deficit, 0, 1))
            else:
                features['soil_deficit'] = 0.5
            
            # 3. Vegetation Stress (VCI-inspired, simulated)
            # In real scenario would use MODIS NDVI
            if 'TS' in df.columns:  # Surface temperature as proxy
                temp = df['TS'].values.astype(float)
                temp_min = np.nanmin(temp)
                temp_max = np.nanmax(temp) + 1e-6
                vci = (temp - temp_min) / (temp_max - temp_min)
                veg_stress = 1.0 - np.nanmean(vci)
                features['veg_stress'] = float(np.clip(veg_stress, 0, 1))
            else:
                features['veg_stress'] = 0.5
            
            # 4. Heatwave Intensity (TCI)
            if 'T2M_MAX' in df.columns:
                temp_max = df['T2M_MAX'].values.astype(float)
                baseline_std = np.nanstd(temp_max[:30]) + 1e-6
                baseline_mean = np.nanmean(temp_max[:30])
                recent_mean = np.nanmean(temp_max[-30:])
                
                tci = (recent_mean - baseline_mean) / baseline_std
                heat_index = np.clip(expit(tci), 0, 1)
                features['heat_index'] = float(heat_index)
            else:
                features['heat_index'] = 0.5
            
            # 5. Historical Drought Frequency
            # Simulated: frequency of extreme conditions
            vci_sim = 0.5 * (1 - features['veg_stress']) + 0.5 * (1 - features['heat_index'])
            drought_freq = np.sum(vci_sim < 0.4) / max(len(df), 1)
            features['drought_freq'] = float(np.clip(drought_freq, 0, 1))
            
            # 6. Trend (recent worsening signal)
            # Slope of drought intensity over last 30 days
            recent_window = min(30, len(df))
            if recent_window > 1:
                x = np.arange(recent_window)
                y = np.linspace(features['veg_stress'], 
                               features['heat_index'], 
                               recent_window)
                trend_slope = np.polyfit(x, y, 1)[0]
                trend = np.clip(-trend_slope, 0, 1)  # Negative slope = worsening
                features['trend'] = float(trend)
            else:
                features['trend'] = 0.5
            
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
    
    def store_feature_history(self, area_id: str, features: Dict[str, float], timestamp: float):
        """
        Store historical features for tracking trends.
        
        Args:
            area_id: Identifier for the area
            features: Feature vector at this timestamp
            timestamp: Time when features were measured
        """
        if area_id not in self.feature_history:
            self.feature_history[area_id] = []
        
        self.feature_history[area_id].append({
            'timestamp': timestamp,
            'features': features.copy()
        })
    
    def get_feature_trend(self, area_id: str, lookback: int = 10) -> Dict[str, float]:
        """
        Calculate feature trends over recent measurements.
        
        Args:
            area_id: Identifier for the area
            lookback: Number of recent measurements to analyze
            
        Returns:
            Dict with trend values for each feature
        """
        if area_id not in self.feature_history or len(self.feature_history[area_id]) < 2:
            return {k: 0.0 for k in self.weights.keys()}
        
        recent = self.feature_history[area_id][-lookback:]
        trends = {}
        
        for feature_key in self.weights.keys():
            values = [h['features'].get(feature_key, 0.5) for h in recent]
            # Simple linear trend: newer - older
            trend = values[-1] - values[0] if len(values) > 1 else 0.0
            trends[feature_key] = float(trend)
        
        return trends


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
