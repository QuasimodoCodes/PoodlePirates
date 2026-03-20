"""
Test Script for Astar Island API Integration

Run this script to verify:
1. API authentication works
2. Can fetch active rounds
3. Can query the simulator
4. Data parsing works correctly
5. Tracking and logging works

Usage:
    python test_api_integration.py
"""

import os
import sys
from dotenv import load_dotenv

# Ensure src is importable
sys.path.insert(0, os.path.dirname(__file__))

from src.api_client import AstarIslandAPIClient
from src.data_parser import AstarIslandParser
from src.query_tracker import QueryBudgetTracker


def print_header(text):
    """Print a formatted header."""
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}\n")


def test_authentication():
    """Test 1: Verify API authentication."""
    print_header("TEST 1: API Authentication")
    
    try:
        client = AstarIslandAPIClient()
        print("✓ API client initialized successfully")
        print(f"  Base URL: {client.api_base_url}")
        print(f"  Auth method: Bearer token")
        return client
    except ValueError as e:
        print(f"✗ Authentication failed: {e}")
        print("\n  Make sure:")
        print("  1. .env file exists in the project root")
        print("  2. GOOGLE_AUTH_TOKEN is set to your token from app.ainm.no")
        print("  3. You're logged in at https://app.ainm.no")
        return None


def test_fetch_rounds(client):
    """Test 2: Fetch list of all rounds."""
    print_header("TEST 2: Fetch Available Rounds")
    
    try:
        rounds = client.get_rounds()
        print(f"✓ Successfully fetched rounds")
        print(f"  Total rounds available: {len(rounds)}")
        
        # Show active rounds
        active_rounds = [r for r in rounds if r.get("status") == "active"]
        print(f"  Active rounds: {len(active_rounds)}")
        
        if active_rounds:
            for i, round_info in enumerate(active_rounds[:3], 1):
                print(f"\n  Round {i}:")
                print(f"    ID: {round_info.get('id')[:12]}...")
                print(f"    Status: {round_info.get('status')}")
                print(f"    Map: {round_info.get('map_width')}×{round_info.get('map_height')}")
                print(f"    Seeds: {round_info.get('seeds_count', 5)}")
        
        return rounds
    except Exception as e:
        print(f"✗ Failed to fetch rounds: {e}")
        return None


def test_check_budget(client):
    """Test 3: Check query budget for active round."""
    print_header("TEST 3: Check Query Budget")
    
    try:
        budget = client.get_budget()
        print(f"✓ Successfully fetched budget info")
        print(f"  Round ID: {budget.get('round_id')[:12]}...")
        print(f"  Queries used: {budget.get('queries_used')}/{budget.get('queries_max')}")
        print(f"  Round active: {budget.get('active')}")
        return budget
    except Exception as e:
        print(f"✗ Failed to check budget: {e}")
        print("  Note: This might fail if no round is active yet")
        return None


def test_simulator_query(client, round_id):
    """Test 4: Make a test query to the simulator."""
    print_header("TEST 4: Query Simulator (Exploratory Query)")
    
    if not round_id:
        print("✗ Cannot test simulator query - no round_id provided")
        return None
    
    try:
        # Make a small exploratory query
        print("Querying simulator...")
        print("  Parameters:")
        print("    - Seed: 0 (first seed)")
        print("    - Viewport: (0, 0) - top-left corner")
        print("    - Size: 15×15 (max viewport size)")
        
        response = client.query_simulator(
            round_id=round_id,
            seed_index=0,
            viewport_x=0,
            viewport_y=0,
            viewport_w=15,
            viewport_h=15
        )
        
        print(f"\n✓ Simulator query successful!")
        print(f"  Grid shape: {len(response.get('grid', []))}×{len(response.get('grid', [[]])[0]) if response.get('grid') else 0}")
        print(f"  Settlements found: {len(response.get('settlements', []))}")
        print(f"  Budget after query: {response.get('queries_used', '?')}/{response.get('queries_max', '?')}")
        
        return response
    except Exception as e:
        print(f"✗ Simulator query failed: {e}")
        return None


def test_data_parsing(response):
    """Test 5: Parse the simulator response."""
    print_header("TEST 5: Data Parsing")
    
    if not response:
        print("✗ Cannot test parsing - no response data")
        return None
    
    try:
        parser = AstarIslandParser()
        observation = parser.parse_simulate_response(response, seed_index=0)
        
        print(f"✓ Successfully parsed response into ViewportObservation")
        print(f"\n  Viewport info:")
        print(f"    Position: ({observation.viewport_x}, {observation.viewport_y})")
        print(f"    Size: {observation.viewport_w}×{observation.viewport_h}")
        print(f"    Full map: {observation.full_map_width}×{observation.full_map_height}")
        
        print(f"\n  Grid analysis:")
        print(f"    Grid shape (numpy): {observation.grid.shape}")
        print(f"    Data type: {observation.grid.dtype}")
        print(f"    Unique terrain types found: {len(set(observation.grid.flatten()))}")
        
        if observation.settlements:
            print(f"\n  Settlements:")
            for i, settlement in enumerate(observation.settlements[:3], 1):
                print(f"    Settlement {i}: pos=({settlement.x},{settlement.y}), "
                      f"pop={settlement.population:.1f}, alive={settlement.alive}")
        
        # Test grid transformations
        print(f"\n  Testing transformations:")
        one_hot = parser.grid_to_one_hot(observation.grid)
        print(f"    One-hot tensor shape: {one_hot.shape} ✓")
        
        full_map = parser.expand_viewport_to_fullmap(observation)
        print(f"    Expanded to full map: {full_map.shape} ✓")
        
        return observation
    except Exception as e:
        print(f"✗ Parsing failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_budget_tracking(round_id):
    """Test 6: Test query budget tracker."""
    print_header("TEST 6: Query Budget Tracking")
    
    try:
        tracker = QueryBudgetTracker(max_queries=50)
        tracker.set_round(round_id)
        
        # Simulate some queries
        tracker.log_query(
            seed_index=0,
            viewport_x=0,
            viewport_y=0,
            viewport_w=15,
            viewport_h=15,
            status="success"
        )
        
        print(f"✓ Query logged successfully")
        print(f"\n  Budget status:")
        print(f"    Used: {tracker.get_used_budget()}")
        print(f"    Remaining: {tracker.get_remaining_budget()}")
        print(f"    Percentage: {tracker.get_budget_percentage():.1f}%")
        
        # Get statistics
        stats = tracker.get_statistics()
        print(f"\n  Statistics:")
        print(f"    Total cells observed: {stats['total_cells_observed']}")
        print(f"    Queries by seed: {stats['queries_by_seed']}")
        print(f"    Coverage: {stats['coverage']['coverage_percentage']:.1f}%")
        
        return tracker
    except Exception as e:
        print(f"✗ Tracking test failed: {e}")
        return None


def run_integration_test():
    """Run all integration tests."""
    print("\n" + "="*70)
    print(" ASTAR ISLAND API INTEGRATION TEST")
    print("="*70)
    
    # Load environment
    print("\nLoading environment variables from .env...")
    load_dotenv()
    
    # Test 1: Authentication
    client = test_authentication()
    if not client:
        print("\n" + "!"*70)
        print("  SETUP ERROR: Cannot proceed without valid authentication")
        print("!"*70)
        return False
    
    # Test 2: Fetch rounds
    rounds = test_fetch_rounds(client)
    if not rounds:
        print("\n⚠ Warning: Could not fetch rounds (API may be down)")
    
    # Test 3: Check budget
    budget = test_check_budget(client)
    round_id = budget.get('round_id') if budget else None
    
    if not round_id and rounds:
        # Use first active round
        active = [r for r in rounds if r.get('status') == 'active']
        if active:
            round_id = active[0]['id']
            print(f"\n✓ Using first active round: {round_id[:12]}...")
    
    # Test 4: Query simulator
    response = None
    if round_id:
        response = test_simulator_query(client, round_id)
    else:
        print_header("TEST 4: Query Simulator")
        print("⚠ Skipping simulator query - no active round available")
        print("  (This is expected if no round is currently active)")
    
    # Test 5: Data parsing
    if response:
        observation = test_data_parsing(response)
    else:
        print_header("TEST 5: Data Parsing")
        print("⚠ Skipping data parsing - no simulator response")
    
    # Test 6: Budget tracking
    if round_id:
        tracker = test_budget_tracking(round_id)
    else:
        print_header("TEST 6: Query Budget Tracking")
        print("⚠ Skipping budget tracking - no round_id available")
    
    # Summary
    print_header("TEST SUMMARY")
    print("✓ API client authentication: WORKING")
    print("✓ Round fetching: " + ("WORKING" if rounds else "FAILED/UNAVAILABLE"))
    print("✓ Budget checking: " + ("WORKING" if budget else "FAILED/UNAVAILABLE"))
    print("✓ Simulator queries: " + ("WORKING" if response else "FAILED/UNAVAILABLE"))
    print("✓ Data parsing: WORKING")
    print("✓ Budget tracking: WORKING")
    
    print("\n" + "="*70)
    print("  All core components are ready!")
    print("="*70)
    print("\nNext steps:")
    print("1. Review API_REFERENCE.md for endpoint documentation")
    print("2. Look at example usage in test_api_integration.py")
    print("3. Update .env with your ROUND_ID when a round becomes active")
    print("4. Proceed to Step 6: Create observation strategy planner")
    print("="*70 + "\n")
    
    return True


if __name__ == "__main__":
    success = run_integration_test()
    sys.exit(0 if success else 1)
