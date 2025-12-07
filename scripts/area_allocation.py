#!/usr/bin/env python3
"""
Area Prioritization & Dynamic Drone Allocation

Implements:
- Risk-based area prioritization
- Dynamic drone allocation to areas
- Coverage-driven reallocation
- Auditor drone management

Paper: DroughtCast - Machine Learning Forecast of the United States Drought Monitor
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class DroneRole(Enum):
    """Drone operational role."""
    EXPLORER = "explorer"      # Primary area monitor
    AUDITOR = "auditor"        # Validates faulty sensors
    BACKUP = "backup"          # Emergency reserve
    IDLE = "idle"              # Not assigned


@dataclass
class Area:
    """Farmland area definition."""
    area_id: str
    drought_probability: float
    size_m2: float
    coverage_radius: float
    x_center: float
    y_center: float
    crop_type: str = "unknown"
    priority_rank: int = 0
    
    def __lt__(self, other):
        """Sort areas by drought probability (highest first)."""
        return other.drought_probability < self.drought_probability


@dataclass
class Drone:
    """Drone definition."""
    drone_id: int
    role: DroneRole = DroneRole.IDLE
    assigned_area: Optional[str] = None
    sensor_noise: float = 0.05
    is_operational: bool = True
    reliability_score: float = 1.0
    x_pos: float = 0.0
    y_pos: float = 0.0
    measurements_count: int = 0


@dataclass
class AllocationResult:
    """Result of drone allocation."""
    allocations: Dict[str, List[int]]  # area_id -> [drone_ids]
    reserve_drones: List[int]
    auditor_assignments: Dict[int, str]  # auditor_id -> area_id
    unassigned_drones: List[int]
    allocation_summary: Dict = field(default_factory=dict)


class AreaPrioritizer:
    """
    Prioritize areas based on drought risk and coverage needs.
    """
    
    def __init__(self):
        """Initialize area prioritizer."""
        self.area_history = {}
    
    def rank_areas_by_risk(self, areas: List[Area]) -> List[Area]:
        """
        Rank areas by drought probability (highest risk first).
        
        Args:
            areas: List of Area objects
            
        Returns:
            Sorted list of areas (highest probability first)
        """
        sorted_areas = sorted(areas, key=lambda a: a.drought_probability, reverse=True)
        
        # Assign priority ranks
        for rank, area in enumerate(sorted_areas):
            area.priority_rank = rank
        
        return sorted_areas
    
    def categorize_areas_by_risk_level(self, areas: List[Area]) -> Dict[str, List[Area]]:
        """
        Categorize areas into risk levels: HIGH (>70%), MEDIUM (40-70%), LOW (<40%).
        
        Args:
            areas: List of Area objects
            
        Returns:
            Dict with keys 'high', 'medium', 'low' containing sorted area lists
        """
        high_risk = []
        medium_risk = []
        low_risk = []
        
        for area in areas:
            if area.drought_probability > 0.70:
                high_risk.append(area)
            elif area.drought_probability > 0.40:
                medium_risk.append(area)
            else:
                low_risk.append(area)
        
        # Sort each category by probability
        high_risk.sort(key=lambda a: a.drought_probability, reverse=True)
        medium_risk.sort(key=lambda a: a.drought_probability, reverse=True)
        low_risk.sort(key=lambda a: a.drought_probability, reverse=True)
        
        return {
            'high': high_risk,
            'medium': medium_risk,
            'low': low_risk
        }
    
    def calculate_coverage_need(self, area: Area) -> int:
        """
        Calculate recommended number of drones for an area.
        
        Larger, higher-risk areas need more coverage.
        
        Args:
            area: Area object
            
        Returns:
            Recommended drone count (1-5)
        """
        # Base allocation: 1 drone per area
        count = 1
        
        # Add drones for high-risk areas
        if area.drought_probability > 0.75:
            count += 1
        
        # Add drones for large areas
        if area.size_m2 > 10000:
            count += 1
        
        # Maximum 5 drones per area
        return min(count, 5)
    
    def store_area_history(self, area_id: str, probability: float, timestamp: float):
        """
        Store historical probability data for tracking trends.
        
        Args:
            area_id: Area identifier
            probability: Measured drought probability
            timestamp: Measurement time
        """
        if area_id not in self.area_history:
            self.area_history[area_id] = []
        
        self.area_history[area_id].append({
            'timestamp': timestamp,
            'probability': probability
        })
    
    def get_area_trend(self, area_id: str) -> str:
        """
        Get trend description for an area.
        
        Args:
            area_id: Area identifier
            
        Returns:
            "improving", "stable", or "worsening"
        """
        if area_id not in self.area_history or len(self.area_history[area_id]) < 2:
            return "unknown"
        
        history = self.area_history[area_id]
        recent = np.mean([h['probability'] for h in history[-5:]])
        old = np.mean([h['probability'] for h in history[:min(5, len(history))]])
        
        if recent > old + 0.05:
            return "worsening"
        elif recent < old - 0.05:
            return "improving"
        else:
            return "stable"


class DynamicDroneAllocator:
    """
    Dynamically allocate drones to areas based on risk and coverage needs.
    """
    
    def __init__(self, total_drones: int, total_areas: int,
                 min_drones_per_area: int = 1,
                 max_drones_per_area: int = 3,
                 reserve_percentage: float = 0.1):
        """
        Initialize drone allocator.
        
        Args:
            total_drones: Total number of drones in fleet
            total_areas: Total number of areas
            min_drones_per_area: Minimum drones per area
            max_drones_per_area: Maximum drones per area
            reserve_percentage: Percentage of drones to keep in reserve (0-0.3)
        """
        self.total_drones = total_drones
        self.total_areas = total_areas
        self.min_drones_per_area = min_drones_per_area
        self.max_drones_per_area = max_drones_per_area
        self.reserve_percentage = reserve_percentage
        
        self.prioritizer = AreaPrioritizer()
        self.allocation_history = []
    
    def allocate_drones(self, areas: List[Area], drones: List[Drone],
                       auditor_deployments: Dict[str, int] = None) -> AllocationResult:
        """
        Allocate drones to areas based on risk and coverage needs.
        
        Algorithm:
        1. Rank areas by drought probability
        2. Allocate minimum drones to all areas (high to low risk)
        3. Allocate additional drones to high-risk areas
        4. Keep reserve for emergencies
        5. Deploy auditors to areas with suspected sensor faults
        
        Args:
            areas: List of Area objects
            drones: List of Drone objects
            auditor_deployments: Dict mapping area_id to required auditor count
            
        Returns:
            AllocationResult object
        """
        if auditor_deployments is None:
            auditor_deployments = {}
        
        # Initialize allocation structure
        allocations: Dict[str, List[int]] = {area.area_id: [] for area in areas}
        reserve_drones = []
        auditor_assignments = {}
        unassigned_drones = [d.drone_id for d in drones if d.is_operational]
        
        # Rank areas by drought probability
        ranked_areas = self.prioritizer.rank_areas_by_risk(areas)
        
        # Calculate reserves
        reserve_count = max(1, int(self.total_drones * self.reserve_percentage))
        available_for_allocation = len(unassigned_drones) - reserve_count
        
        # PHASE 1: Allocate minimum drones to all areas (coverage guarantee)
        for area in ranked_areas:
            if available_for_allocation > 0:
                # Allocate one explorer drone to each area
                drone_id = unassigned_drones.pop(0)
                allocations[area.area_id].append(drone_id)
                available_for_allocation -= 1
        
        # PHASE 2: Deploy auditors to suspect areas
        for area_id, auditor_count in auditor_deployments.items():
            for _ in range(auditor_count):
                if unassigned_drones:
                    auditor_id = unassigned_drones.pop(0)
                    allocations[area_id].append(auditor_id)
                    auditor_assignments[auditor_id] = area_id
                    available_for_allocation -= 1
        
        # PHASE 3: Allocate additional drones to high-risk areas
        high_risk_categories = self.prioritizer.categorize_areas_by_risk_level(ranked_areas)
        
        for area in high_risk_categories['high']:
            while (len(allocations[area.area_id]) < self.max_drones_per_area and
                   available_for_allocation > 0):
                drone_id = unassigned_drones.pop(0)
                allocations[area.area_id].append(drone_id)
                available_for_allocation -= 1
        
        # PHASE 4: Allocate remaining to medium-risk areas (if space available)
        for area in high_risk_categories['medium']:
            if (len(allocations[area.area_id]) < self.max_drones_per_area - 1 and
                available_for_allocation > 0):
                drone_id = unassigned_drones.pop(0)
                allocations[area.area_id].append(drone_id)
                available_for_allocation -= 1
        
        # PHASE 5: Reserve remaining drones
        reserve_drones = unassigned_drones[:reserve_count]
        unassigned_drones = unassigned_drones[reserve_count:]
        
        # Create allocation summary
        summary = self._generate_summary(ranked_areas, allocations, reserve_drones)
        
        # Store in history
        self.allocation_history.append(summary)
        
        return AllocationResult(
            allocations=allocations,
            reserve_drones=reserve_drones,
            auditor_assignments=auditor_assignments,
            unassigned_drones=unassigned_drones,
            allocation_summary=summary
        )
    
    def _generate_summary(self, ranked_areas: List[Area], 
                         allocations: Dict[str, List[int]], 
                         reserve_drones: List[int]) -> Dict:
        """
        Generate allocation summary statistics.
        
        Args:
            ranked_areas: Ranked list of areas
            allocations: Allocation results
            reserve_drones: Reserve drone IDs
            
        Returns:
            Summary dictionary
        """
        total_allocated = sum(len(d) for d in allocations.values())
        
        return {
            'total_allocated': total_allocated,
            'total_reserves': len(reserve_drones),
            'areas_covered': sum(1 for d in allocations.values() if len(d) > 0),
            'total_areas': len(ranked_areas),
            'high_risk_areas': sum(1 for a in ranked_areas if a.drought_probability > 0.70),
            'medium_risk_areas': sum(1 for a in ranked_areas if 0.40 < a.drought_probability <= 0.70),
            'low_risk_areas': sum(1 for a in ranked_areas if a.drought_probability <= 0.40),
            'allocation_table': [
                {
                    'area_id': area.area_id,
                    'probability': area.drought_probability,
                    'priority': area.priority_rank,
                    'drones_assigned': len(allocations[area.area_id]),
                    'drone_ids': allocations[area.area_id]
                }
                for area in ranked_areas
            ]
        }
    
    def reallocate_on_fault(self, faulty_area_id: str, 
                           allocations: Dict[str, List[int]],
                           reserve_drones: List[int]) -> Optional[int]:
        """
        Reallocate drones when a sensor fault is detected.
        
        Deploy an auditor from reserve to the faulty area.
        
        Args:
            faulty_area_id: Area with faulty sensor
            allocations: Current allocations
            reserve_drones: Available reserve drones
            
        Returns:
            ID of deployed auditor, or None if no reserves available
        """
        if not reserve_drones:
            print(f"[WARNING] No reserve drones available for {faulty_area_id}")
            return None
        
        # Deploy best available reserve drone
        auditor_id = reserve_drones.pop(0)
        allocations[faulty_area_id].append(auditor_id)
        
        print(f"[REALLOCATION] Auditor drone {auditor_id} deployed to {faulty_area_id}")
        
        return auditor_id
    
    def get_allocation_stats(self) -> Dict:
        """
        Get statistics about allocation patterns.
        
        Returns:
            Dictionary with allocation statistics
        """
        if not self.allocation_history:
            return {}
        
        recent = self.allocation_history[-1]
        
        stats = {
            'timestamp': 'latest',
            'total_drones_allocated': recent['total_allocated'],
            'drones_in_reserve': recent['total_reserves'],
            'areas_covered': recent['areas_covered'],
            'coverage_percentage': (recent['areas_covered'] / recent['total_areas']) * 100,
            'risk_distribution': {
                'high': recent['high_risk_areas'],
                'medium': recent['medium_risk_areas'],
                'low': recent['low_risk_areas']
            },
            'avg_drones_per_area': recent['total_allocated'] / max(recent['areas_covered'], 1)
        }
        
        return stats


def test_allocation():
    """Test area prioritization and drone allocation."""
    print("=" * 60)
    print("Testing Area Prioritization & Drone Allocation")
    print("=" * 60)
    
    # Create test areas with different risk levels
    areas = [
        Area(area_id="area_1", drought_probability=0.85, size_m2=12000, 
             coverage_radius=60, x_center=-12, y_center=9, crop_type="Wheat"),
        Area(area_id="area_2", drought_probability=0.65, size_m2=8000, 
             coverage_radius=50, x_center=-2, y_center=11.5, crop_type="Corn"),
        Area(area_id="area_3", drought_probability=0.45, size_m2=5000, 
             coverage_radius=40, x_center=11, y_center=7.5, crop_type="Soybean"),
        Area(area_id="area_4", drought_probability=0.72, size_m2=9000, 
             coverage_radius=55, x_center=-11, y_center=1.5, crop_type="Wheat"),
        Area(area_id="area_5", drought_probability=0.35, size_m2=7000, 
             coverage_radius=45, x_center=1, y_center=-1, crop_type="Barley"),
    ]
    
    # Create test drones
    drones = [Drone(drone_id=i, sensor_noise=0.05) for i in range(10)]
    
    # Test 1: Area prioritization
    print("\n[Test 1] Area Risk Ranking:")
    prioritizer = AreaPrioritizer()
    ranked = prioritizer.rank_areas_by_risk(areas)
    
    for area in ranked:
        risk_level = "HIGH" if area.drought_probability > 0.70 else \
                    "MEDIUM" if area.drought_probability > 0.40 else "LOW"
        print(f"  Rank {area.priority_rank + 1}: {area.area_id} - "
              f"Risk: {area.drought_probability:.3f} ({risk_level})")
    
    # Test 2: Coverage needs
    print("\n[Test 2] Coverage Needs per Area:")
    for area in ranked:
        coverage = prioritizer.calculate_coverage_need(area)
        print(f"  {area.area_id}: {coverage} drone(s) recommended")
    
    # Test 3: Risk categorization
    print("\n[Test 3] Risk Categorization:")
    categories = prioritizer.categorize_areas_by_risk_level(areas)
    print(f"  HIGH risk ({len(categories['high'])} areas):")
    for a in categories['high']:
        print(f"    - {a.area_id}: {a.drought_probability:.3f}")
    print(f"  MEDIUM risk ({len(categories['medium'])} areas):")
    for a in categories['medium']:
        print(f"    - {a.area_id}: {a.drought_probability:.3f}")
    print(f"  LOW risk ({len(categories['low'])} areas):")
    for a in categories['low']:
        print(f"    - {a.area_id}: {a.drought_probability:.3f}")
    
    # Test 4: Dynamic allocation
    print("\n[Test 4] Dynamic Drone Allocation:")
    allocator = DynamicDroneAllocator(
        total_drones=10,
        total_areas=5,
        min_drones_per_area=1,
        max_drones_per_area=2,
        reserve_percentage=0.1
    )
    
    result = allocator.allocate_drones(areas, drones)
    
    print(f"  Total drones allocated: {result.allocation_summary['total_allocated']}")
    print(f"  Drones in reserve: {result.allocation_summary['total_reserves']}")
    print(f"  Areas covered: {result.allocation_summary['areas_covered']}/{result.allocation_summary['total_areas']}")
    
    print("\n  Allocation table:")
    for entry in result.allocation_summary['allocation_table']:
        print(f"    {entry['area_id']} (Risk: {entry['probability']:.3f}, Priority: {entry['priority']})")
        print(f"      Drones: {entry['drone_ids']}")
    
    # Test 5: Reallocation on fault
    print("\n[Test 5] Reallocation on Sensor Fault:")
    print(f"  Reserve drones before: {result.reserve_drones}")
    
    auditor_id = allocator.reallocate_on_fault(
        "area_1",
        result.allocations,
        result.reserve_drones
    )
    
    print(f"  Auditor deployed: {auditor_id}")
    print(f"  Reserve drones after: {result.reserve_drones}")
    print(f"  area_1 drones now: {result.allocations['area_1']}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_allocation()
