"""
Data Parser for Astar Island Simulator Responses

Parses API responses and converts them to convenient data structures
and numpy arrays for analysis and model building.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass


# Cell type constants
TERRAIN_CLASSES = {
    0: "Empty",
    1: "Settlement", 
    2: "Port",
    3: "Ruin",
    4: "Forest",
    5: "Mountain",
    10: "Ocean",
    11: "Plains"
}

TERRAIN_TO_CLASS = {
    "Empty": 0,
    "Settlement": 1,
    "Port": 2,
    "Ruin": 3,
    "Forest": 4,
    "Mountain": 5,
    "Ocean": 10,
    "Plains": 11
}


@dataclass
class Settlement:
    """Represents a settlement in the simulation."""
    x: int
    y: int
    population: float
    food: float
    wealth: float
    defense: float
    has_port: bool
    alive: bool
    owner_id: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "x": self.x,
            "y": self.y,
            "population": self.population,
            "food": self.food,
            "wealth": self.wealth,
            "defense": self.defense,
            "has_port": self.has_port,
            "alive": self.alive,
            "owner_id": self.owner_id,
        }


@dataclass
class ViewportObservation:
    """Represents one simulator query result."""
    seed_index: int
    viewport_x: int
    viewport_y: int
    viewport_w: int
    viewport_h: int
    grid: np.ndarray           # viewport_h × viewport_w
    settlements: List[Settlement]
    full_map_width: int = 40
    full_map_height: int = 40
    queries_used: int = 0
    queries_max: int = 50
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "seed_index": self.seed_index,
            "viewport": {
                "x": self.viewport_x,
                "y": self.viewport_y,
                "w": self.viewport_w,
                "h": self.viewport_h,
            },
            "grid": self.grid.tolist(),
            "settlements": [s.to_dict() for s in self.settlements],
            "full_map": {
                "width": self.full_map_width,
                "height": self.full_map_height,
            },
            "budget": {
                "used": self.queries_used,
                "max": self.queries_max,
            }
        }


class AstarIslandParser:
    """
    Parser for Astar Island API responses.
    Converts raw API data into usable Python objects and numpy arrays.
    """
    
    @staticmethod
    def parse_simulate_response(response: Dict[str, Any], seed_index: int) -> ViewportObservation:
        """
        Parse a POST /simulate response into a ViewportObservation object.
        
        Args:
            response: Raw API response from query_simulator()
            seed_index: The seed index that was queried (0-4)
        
        Returns:
            ViewportObservation with parsed grid, settlements, and metadata
        """
        # Parse grid
        grid_data = response.get("grid", [])
        grid = np.array(grid_data, dtype=np.int32)
        
        # Parse settlements
        settlements = []
        for settlement_data in response.get("settlements", []):
            settlement = Settlement(
                x=settlement_data.get("x"),
                y=settlement_data.get("y"),
                population=settlement_data.get("population", 0.0),
                food=settlement_data.get("food", 0.0),
                wealth=settlement_data.get("wealth", 0.0),
                defense=settlement_data.get("defense", 0.0),
                has_port=settlement_data.get("has_port", False),
                alive=settlement_data.get("alive", True),
                owner_id=settlement_data.get("owner_id", -1),
            )
            settlements.append(settlement)
        
        # Parse viewport info
        viewport = response.get("viewport", {})
        
        observation = ViewportObservation(
            seed_index=seed_index,
            viewport_x=viewport.get("x", 0),
            viewport_y=viewport.get("y", 0),
            viewport_w=viewport.get("w", grid.shape[1]),
            viewport_h=viewport.get("h", grid.shape[0]),
            grid=grid,
            settlements=settlements,
            full_map_width=response.get("width", 40),
            full_map_height=response.get("height", 40),
            queries_used=response.get("queries_used", 0),
            queries_max=response.get("queries_max", 50),
        )
        
        return observation
    
    @staticmethod
    def grid_to_one_hot(grid: np.ndarray, num_classes: int = 6) -> np.ndarray:
        """
        Convert a grid of class indices to one-hot encoding.
        
        Args:
            grid: 2D array of class indices (H × W)
            num_classes: Number of classes (default 6)
        
        Returns:
            3D one-hot array (H × W × num_classes)
            where array[y, x, c] = 1 if grid[y, x] == c, else 0
        """
        h, w = grid.shape
        one_hot = np.zeros((h, w, num_classes), dtype=np.float32)
        
        for y in range(h):
            for x in range(w):
                class_idx = grid[y, x]
                if 0 <= class_idx < num_classes:
                    one_hot[y, x, class_idx] = 1.0
        
        return one_hot
    
    @staticmethod
    def grid_to_probability(grid: np.ndarray, num_classes: int = 6) -> np.ndarray:
        """
        Convert a grid of class indices to probability distribution.
        
        Simple version: each cell gets 1.0 for its class, 0.0 for others.
        For more sophisticated methods, use your ML model instead.
        
        Args:
            grid: 2D array of class indices (H × W)
            num_classes: Number of classes (default 6)
        
        Returns:
            3D probability array (H × W × num_classes)
        """
        return AstarIslandParser.grid_to_one_hot(grid, num_classes)
    
    @staticmethod
    def expand_viewport_to_fullmap(
        viewport_observation: ViewportObservation,
        default_class: int = 0,
    ) -> np.ndarray:
        """
        Expand a viewport observation to a full 40×40 map grid.
        Unmapped regions are filled with default_class.
        
        Args:
            viewport_observation: The parsed viewport data
            default_class: What class to use for unknown cells (default 0=Empty)
        
        Returns:
            Full 40×40 grid with viewport data placed at correct position,
            rest filled with default_class
        """
        full_grid = np.full((40, 40), default_class, dtype=np.int32)
        
        # Place viewport data at correct position
        vp = viewport_observation
        y_start = vp.viewport_y
        y_end = vp.viewport_y + vp.viewport_h
        x_start = vp.viewport_x
        x_end = vp.viewport_x + vp.viewport_w
        
        full_grid[y_start:y_end, x_start:x_end] = vp.grid
        
        return full_grid
    
    @staticmethod
    def create_settlement_heatmap(
        viewport_observation: ViewportObservation,
        map_size: Tuple[int, int] = (40, 40),
    ) -> np.ndarray:
        """
        Create a heatmap of settlement strength/development.
        
        Args:
            viewport_observation: The parsed viewport data
            map_size: Full map dimensions (default 40×40)
        
        Returns:
            2D array (H × W) with settlement values
            - 0: No settlement
            - >0: Settlement strength based on population
        """
        heatmap = np.zeros(map_size, dtype=np.float32)
        
        for settlement in viewport_observation.settlements:
            # Only map settlements that are within the map bounds
            if 0 <= settlement.x < map_size[1] and 0 <= settlement.y < map_size[0]:
                # Use population as settlement strength indicator
                heatmap[settlement.y, settlement.x] = settlement.population
        
        return heatmap
    
    @staticmethod
    def extract_terrain_statistics(observations: List[ViewportObservation]) -> Dict[int, int]:
        """
        Count occurrences of each terrain class across multiple observations.
        
        Useful for understanding the distribution of terrain types.
        
        Args:
            observations: List of ViewportObservation objects
        
        Returns:
            Dictionary mapping class index -> count
        """
        class_counts = {i: 0 for i in range(6)}
        total_cells = 0
        
        for obs in observations:
            unique, counts = np.unique(obs.grid, return_counts=True)
            for terrain_class, count in zip(unique, counts):
                if terrain_class in class_counts:
                    class_counts[terrain_class] += count
            total_cells += obs.grid.size
        
        # Optionally normalize to percentages
        percentages = {k: (v / total_cells * 100) if total_cells > 0 else 0 
                      for k, v in class_counts.items()}
        
        return {"counts": class_counts, "percentages": percentages, "total_cells": total_cells}
    
    @staticmethod
    def analyze_viewport_coverage(observations: List[ViewportObservation]) -> Dict[str, Any]:
        """
        Analyze which parts of the map have been observed.
        
        Args:
            observations: List of ViewportObservation objects
        
        Returns:
            Coverage statistics including:
            - coverage_map: 2D boolean array indicating observed cells
            - coverage_percentage: Percent of map observed
            - viewports: List of observed viewport regions
        """
        coverage_map = np.zeros((40, 40), dtype=np.bool_)
        viewport_regions = []
        
        for obs in observations:
            y_start = obs.viewport_y
            y_end = obs.viewport_y + obs.viewport_h
            x_start = obs.viewport_x
            x_end = obs.viewport_x + obs.viewport_w
            
            coverage_map[y_start:y_end, x_start:x_end] = True
            
            viewport_regions.append({
                "x": obs.viewport_x,
                "y": obs.viewport_y,
                "w": obs.viewport_w,
                "h": obs.viewport_h,
                "seed": obs.seed_index,
            })
        
        covered_cells = np.sum(coverage_map)
        total_cells = 40 * 40
        coverage_pct = (covered_cells / total_cells) * 100
        
        return {
            "coverage_map": coverage_map,
            "coverage_percentage": coverage_pct,
            "covered_cells": covered_cells,
            "total_cells": total_cells,
            "viewports": viewport_regions,
        }
    
    @staticmethod
    def save_observation(observation: ViewportObservation, filepath: str) -> None:
        """Save observation to JSON file."""
        import json
        with open(filepath, 'w') as f:
            json.dump(observation.to_dict(), f, indent=2)
    
    @staticmethod
    def load_observation(filepath: str) -> ViewportObservation:
        """Load observation from JSON file."""
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Reconstruct ViewportObservation from dict
        settlements = [
            Settlement(**s) for s in data.get("settlements", [])
        ]
        
        return ViewportObservation(
            seed_index=data["seed_index"],
            viewport_x=data["viewport"]["x"],
            viewport_y=data["viewport"]["y"],
            viewport_w=data["viewport"]["w"],
            viewport_h=data["viewport"]["h"],
            grid=np.array(data["grid"], dtype=np.int32),
            settlements=settlements,
            full_map_width=data["full_map"]["width"],
            full_map_height=data["full_map"]["height"],
            queries_used=data["budget"]["used"],
            queries_max=data["budget"]["max"],
        )
