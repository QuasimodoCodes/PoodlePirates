"""
Query Budget Tracker for Astar Island

Tracks all simulator queries, budget usage, and logs for debugging and optimization.
Helps ensure queries are used strategically within the 50-query limit.
"""

import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class QueryLog:
    """Record of a single query to the simulator."""
    query_number: int
    timestamp: str
    round_id: str
    seed_index: int
    viewport_x: int
    viewport_y: int
    viewport_w: int
    viewport_h: int
    status: str  # "success", "failed", "rate_limited"
    error_message: Optional[str] = None
    cells_observed: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class QueryBudgetTracker:
    """
    Tracks query budget usage and maintains detailed logs of all queries.
    
    Features:
    - Log every query with coordinates and timestamp
    - Track budget (remaining/used/max)
    - Persist logs to JSON file
    - Calculate coverage statistics
    - Optimize query placement
    """
    
    def __init__(self, max_queries: int = 50, log_file: Optional[str] = None):
        """
        Initialize the tracker.
        
        Args:
            max_queries: Total query budget (default 50)
            log_file: Path to save query logs (default: data/query_log.json)
        """
        self.max_queries = max_queries
        self.queries_used = 0
        self.log_file = Path(log_file or "data/query_log.json")
        self.logs: List[QueryLog] = []
        self.round_id: Optional[str] = None
    
    def set_round(self, round_id: str) -> None:
        """Set the current round ID."""
        self.round_id = round_id
        print(f"✓ Tracker now logging for round: {round_id[:8]}...")
    
    def log_query(
        self,
        seed_index: int,
        viewport_x: int,
        viewport_y: int,
        viewport_w: int,
        viewport_h: int,
        status: str = "success",
        error_message: Optional[str] = None,
    ) -> None:
        """
        Log a query to the simulator.
        
        Args:
            seed_index: Seed index (0-4)
            viewport_x: Viewport X coordinate
            viewport_y: Viewport Y coordinate
            viewport_w: Viewport width
            viewport_h: Viewport height
            status: Query status ("success", "failed", "rate_limited")
            error_message: Error message if status != "success"
        """
        if status == "success":
            self.queries_used += 1
        
        cells_observed = viewport_w * viewport_h
        
        log_entry = QueryLog(
            query_number=self.queries_used,
            timestamp=datetime.now().isoformat(),
            round_id=self.round_id or "unknown",
            seed_index=seed_index,
            viewport_x=viewport_x,
            viewport_y=viewport_y,
            viewport_w=viewport_w,
            viewport_h=viewport_h,
            status=status,
            error_message=error_message,
            cells_observed=cells_observed,
        )
        
        self.logs.append(log_entry)
        
        # Print summary
        remaining = self.max_queries - self.queries_used
        status_symbol = "✓" if status == "success" else "✗"
        print(f"{status_symbol} Query #{self.queries_used}/{self.max_queries} | "
              f"Seed {seed_index} | ({viewport_x},{viewport_y}) {viewport_w}×{viewport_h} | "
              f"{remaining} remaining")
    
    def get_remaining_budget(self) -> int:
        """Get remaining queries."""
        return self.max_queries - self.queries_used
    
    def get_used_budget(self) -> int:
        """Get used queries."""
        return self.queries_used
    
    def get_budget_percentage(self) -> float:
        """Get percentage of budget used."""
        return (self.queries_used / self.max_queries) * 100
    
    def is_budget_exhausted(self) -> bool:
        """Check if all queries are used."""
        return self.queries_used >= self.max_queries
    
    def print_budget_summary(self) -> None:
        """Print budget summary."""
        remaining = self.get_remaining_budget()
        percentage = self.get_budget_percentage()
        print(f"\n{'='*60}")
        print(f"BUDGET SUMMARY")
        print(f"{'='*60}")
        print(f"Used:      {self.queries_used:2d}/{self.max_queries}")
        print(f"Remaining: {remaining:2d}/{self.max_queries}")
        print(f"Progress:  {percentage:5.1f}% complete")
        print(f"{'='*60}\n")
    
    def get_queries_by_seed(self) -> Dict[int, int]:
        """Count queries per seed."""
        counts = {i: 0 for i in range(5)}
        for log in self.logs:
            if log.status == "success":
                counts[log.seed_index] += 1
        return counts
    
    def get_coverage_map(self) -> Dict[str, Any]:
        """
        Analyze which regions of the 40×40 map have been observed.
        
        Returns:
            Dictionary with coverage statistics
        """
        import numpy as np
        
        coverage = np.zeros((40, 40), dtype=np.bool_)
        viewports = []
        
        for log in self.logs:
            if log.status == "success":
                y_start = log.viewport_y
                y_end = min(log.viewport_y + log.viewport_h, 40)
                x_start = log.viewport_x
                x_end = min(log.viewport_x + log.viewport_w, 40)
                
                coverage[y_start:y_end, x_start:x_end] = True
                
                viewports.append({
                    "x": log.viewport_x,
                    "y": log.viewport_y,
                    "w": log.viewport_w,
                    "h": log.viewport_h,
                    "seed": log.seed_index,
                })
        
        covered = np.sum(coverage)
        total = 40 * 40
        coverage_pct = (covered / total) * 100
        
        return {
            "coverage_map": coverage.astype(int).tolist(),
            "covered_cells": int(covered),
            "total_cells": total,
            "coverage_percentage": coverage_pct,
            "viewports": viewports,
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detailed query statistics."""
        queries_by_seed = self.get_queries_by_seed()
        coverage = self.get_coverage_map()
        
        total_cells_observed = sum(log.cells_observed for log in self.logs 
                                  if log.status == "success")
        
        return {
            "total_queries": self.queries_used,
            "max_queries": self.max_queries,
            "remaining_queries": self.get_remaining_budget(),
            "budget_percentage_used": self.get_budget_percentage(),
            "queries_by_seed": queries_by_seed,
            "total_cells_observed": total_cells_observed,
            "failed_queries": sum(1 for log in self.logs if log.status != "success"),
            "coverage": coverage,
        }
    
    def reset(self) -> None:
        """Reset tracker (start new round)."""
        self.queries_used = 0
        self.logs = []
        self.round_id = None
        print("✓ Tracker reset for new round")
    
    def save_logs(self, filepath: Optional[str] = None) -> None:
        """
        Save query logs to JSON file.
        
        Args:
            filepath: Custom path to save (default: data/query_log.json)
        """
        filepath = Path(filepath or self.log_file)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        logs_data = {
            "metadata": {
                "total_queries": self.queries_used,
                "max_queries": self.max_queries,
                "round_id": self.round_id,
                "saved_at": datetime.now().isoformat(),
            },
            "logs": [log.to_dict() for log in self.logs],
            "statistics": self.get_statistics(),
        }
        
        with open(filepath, 'w') as f:
            json.dump(logs_data, f, indent=2)
        
        print(f"✓ Logs saved to {filepath}")
    
    def load_logs(self, filepath: Optional[str] = None) -> None:
        """
        Load query logs from JSON file.
        
        Args:
            filepath: Path to load (default: data/query_log.json)
        """
        filepath = Path(filepath or self.log_file)
        
        if not filepath.exists():
            print(f"✗ Log file not found: {filepath}")
            return
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        metadata = data.get("metadata", {})
        self.round_id = metadata.get("round_id")
        self.queries_used = metadata.get("total_queries", 0)
        
        self.logs = [QueryLog(**log_dict) for log_dict in data.get("logs", [])]
        
        print(f"✓ Loaded {len(self.logs)} query logs from {filepath}")
    
    def export_csv(self, filepath: str) -> None:
        """Export logs to CSV for analysis."""
        import csv
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'query_number', 'timestamp', 'seed_index',
                'viewport_x', 'viewport_y', 'viewport_w', 'viewport_h',
                'cells_observed', 'status', 'error_message'
            ])
            writer.writeheader()
            for log in self.logs:
                writer.writerow(log.to_dict())
        
        print(f"✓ Exported {len(self.logs)} queries to {filepath}")
    
    def suggest_next_viewport(self) -> Tuple[int, int, int, int]:
        """
        Suggest next viewport coordinates based on coverage.
        
        Simple strategy: suggest a viewport in the least-covered region.
        
        Returns:
            Tuple of (viewport_x, viewport_y, viewport_w, viewport_h) for next query
        """
        import numpy as np
        
        coverage = np.zeros((40, 40), dtype=int)
        
        # Build coverage map with query counts
        for log in self.logs:
            if log.status == "success":
                y_start = log.viewport_y
                y_end = min(log.viewport_y + log.viewport_h, 40)
                x_start = log.viewport_x
                x_end = min(log.viewport_x + log.viewport_w, 40)
                coverage[y_start:y_end, x_start:x_end] += 1
        
        # Find least-covered region (prefer completely unobserved)
        min_coverage = coverage.min()
        least_covered = np.argwhere(coverage == min_coverage)
        
        if len(least_covered) > 0:
            # Pick a random cell from least-covered
            idx = np.random.randint(0, len(least_covered))
            y, x = least_covered[idx]
            
            # Clamp viewport to map bounds
            vp_x = max(0, min(x, 40 - 15))
            vp_y = max(0, min(y, 40 - 15))
            
            return (vp_x, vp_y, 15, 15)
        
        # Fallback: random viewport
        return (
            np.random.randint(0, 26),  # 0-25 (so w=15 stays in bounds)
            np.random.randint(0, 26),
            15,
            15
        )
    
    def print_detailed_logs(self, limit: Optional[int] = None) -> None:
        """
        Print detailed query logs.
        
        Args:
            limit: Max number of logs to print (None = all)
        """
        logs_to_print = self.logs if limit is None else self.logs[-limit:]
        
        print(f"\n{'='*100}")
        print(f"{'QUERY LOGS':^100}")
        print(f"{'='*100}")
        print(f"{'#':<4} {'Seed':<5} {'X':<4} {'Y':<4} {'W':<3} {'H':<3} {'Status':<10} {'Timestamp':<25}")
        print(f"{'-'*100}")
        
        for log in logs_to_print:
            timestamp = log.timestamp.split('T')[1][:8]  # HH:MM:SS
            print(f"{log.query_number:<4} {log.seed_index:<5} {log.viewport_x:<4} {log.viewport_y:<4} "
                  f"{log.viewport_w:<3} {log.viewport_h:<3} {log.status:<10} {timestamp:<25}")
        
        print(f"{'='*100}\n")
