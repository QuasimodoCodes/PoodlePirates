"""
Astar Island API Client

Handles authentication and core API interactions with the Norse world simulator.
"""

import os
import json
from typing import Dict, Any, Optional, Tuple
from dotenv import load_dotenv
import requests
from datetime import datetime

# Load environment variables
load_dotenv()


class AstarIslandAPIClient:
    """
    Client for interacting with the Astar Island Norse world simulator API.
    
    Handles:
    - Authentication with Google OAuth tokens
    - POST /astar-island/simulate for querying simulator viewport
    - POST /astar-island/predict for submitting predictions
    - Error handling and logging
    """
    
    def __init__(self, api_base_url: Optional[str] = None, auth_token: Optional[str] = None):
        """
        Initialize the API client with authentication.
        
        Args:
            api_base_url: Base URL for the API (default from .env: https://api.ainm.no/astar-island)
            auth_token: Google OAuth token (default from .env: GOOGLE_AUTH_TOKEN)
        
        Raises:
            ValueError: If authentication token is not provided or found in environment
        """
        self.api_base_url = api_base_url or os.getenv("API_BASE_URL", "https://api.ainm.no/astar-island")
        self.auth_token = auth_token or os.getenv("GOOGLE_AUTH_TOKEN")
        
        if not self.auth_token:
            raise ValueError(
                "Google OAuth token not found. "
                "Set GOOGLE_AUTH_TOKEN in .env file or pass auth_token parameter."
            )
        
        # Set up headers with authentication
        self.headers = {
            "Authorization": f"Bearer {self.auth_token}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        # Optional: Load round and team info from .env
        self.round_id = os.getenv("ROUND_ID")
        self.team_id = os.getenv("TEAM_ID")
        
        # Track API calls for logging
        self.call_count = 0
        self.last_call_time = None
        
        print(f"✓ API Client initialized")
        print(f"  Base URL: {self.api_base_url}")
        if self.round_id:
            print(f"  Round ID: {self.round_id}")
        if self.team_id:
            print(f"  Team ID: {self.team_id}")
    
    def _make_request(
        self, 
        endpoint: str, 
        method: str = "POST", 
        payload: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make an authenticated HTTP request to the API.
        
        Args:
            endpoint: API endpoint (e.g., "/simulate", "/predict")
            method: HTTP method (default: "POST")
            payload: Request body as dictionary
            params: Query parameters as dictionary
        
        Returns:
            Parsed JSON response as dictionary
        
        Raises:
            requests.exceptions.RequestException: If request fails
            ValueError: If response status indicates an error
        """
        url = f"{self.api_base_url}{endpoint}"
        
        try:
            if method.upper() == "POST":
                response = requests.post(
                    url,
                    headers=self.headers,
                    json=payload,
                    params=params,
                    timeout=30
                )
            elif method.upper() == "GET":
                response = requests.get(
                    url,
                    headers=self.headers,
                    params=params,
                    timeout=30
                )
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            # Track API calls
            self.call_count += 1
            self.last_call_time = datetime.now()
            
            # Handle errors
            if response.status_code >= 400:
                error_message = f"API Error {response.status_code}: {response.text}"
                raise ValueError(error_message)
            
            return response.json()
        
        except requests.exceptions.Timeout:
            raise requests.exceptions.RequestException("API request timed out (30s)")
        except requests.exceptions.ConnectionError as e:
            raise requests.exceptions.RequestException(f"Failed to connect to API: {str(e)}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON response from API: {response.text}")
    
    def test_connection(self) -> bool:
        """
        Test API connection by making a simple authenticated request.
        Fetches the list of all rounds to verify authentication works.
        
        Returns:
            True if connection successful, raises exception otherwise
        """
        try:
            # Try to make a test request to verify authentication
            response = self._make_request(
                "/rounds",
                method="GET"
            )
            print(f"✓ API Connection test passed (found {len(response)} rounds)")
            return True
        except ValueError as e:
            print(f"✗ API Connection test failed: {str(e)}")
            return False
        except requests.exceptions.RequestException as e:
            print(f"✗ API Connection test failed: {str(e)}")
            return False
    
    def query_simulator(
        self,
        round_id: str,
        seed_index: int,
        viewport_x: int,
        viewport_y: int,
        viewport_w: int = 15,
        viewport_h: int = 15
    ) -> Dict[str, Any]:
        """
        Query the simulator for a viewport observation.
        
        Each call runs one stochastic simulation and reveals a viewport window.
        Costs one query from your 50-query budget.
        
        Args:
            round_id: UUID of the active round
            seed_index: Seed number (0-4) to observe
            viewport_x: Starting X coordinate of viewport (0-39)
            viewport_y: Starting Y coordinate of viewport (0-39)
            viewport_w: Width of viewport (5-15, default 15)
            viewport_h: Height of viewport (5-15, default 15)
        
        Returns:
            Simulator response containing:
            - grid: 2D array (viewport_h × viewport_w) of cell values
            - settlements: List of settlements with stats (x, y, population, food, wealth, defense, etc.)
            - viewport: Confirmed viewport bounds {x, y, w, h}
            - width, height: Full map dimensions (40×40)
            - queries_used, queries_max: Budget tracking
        
        Raises:
            ValueError: If parameters are invalid or API returns error
        
        Cell values:
            0 = Empty (Ocean/Plains)
            1 = Settlement
            2 = Port
            3 = Ruin
            4 = Forest
            5 = Mountain
        """
        # Validate parameters
        if not 0 <= seed_index < 5:
            raise ValueError(f"seed_index must be 0-4, got {seed_index}")
        if not 0 <= viewport_x < 40:
            raise ValueError(f"viewport_x must be 0-39, got {viewport_x}")
        if not 0 <= viewport_y < 40:
            raise ValueError(f"viewport_y must be 0-39, got {viewport_y}")
        if not 5 <= viewport_w <= 15:
            raise ValueError(f"viewport_w must be 5-15, got {viewport_w}")
        if not 5 <= viewport_h <= 15:
            raise ValueError(f"viewport_h must be 5-15, got {viewport_h}")
        if not round_id:
            raise ValueError("round_id is required")
        
        payload = {
            "round_id": round_id,
            "seed_index": seed_index,
            "viewport_x": viewport_x,
            "viewport_y": viewport_y,
            "viewport_w": viewport_w,
            "viewport_h": viewport_h
        }
        
        response = self._make_request("/simulate", method="POST", payload=payload)
        print(f"✓ Query #{self.call_count}: Round {round_id[:8]}..., Seed {seed_index}, "
              f"Viewport ({viewport_x}, {viewport_y}) {viewport_w}×{viewport_h}")
        
        return response
    
    def submit_predictions(
        self,
        round_id: str,
        seed_index: int,
        predictions: Any
    ) -> Dict[str, Any]:
        """
        Submit probability predictions for a given seed.
        
        Resubmitting for the same seed overwrites your previous prediction.
        
        Args:
            round_id: UUID of the round
            seed_index: Seed number (0-4) to submit predictions for
            predictions: H×W×6 array of probability distributions per cell
                        - H = height (40)
                        - W = width (40)
                        - 6 = probability per class
                        - prediction[y][x][class] format
                        - Each cell's 6 probabilities must sum to 1.0 (±0.01 tolerance)
                        - Can be numpy array or nested list
        
        Returns:
            API response: {"status": "accepted", "round_id": "...", "seed_index": 3}
        
        Raises:
            ValueError: If parameters are invalid or API returns error
        
        Class indices:
            0 = Empty (Ocean, Plains)
            1 = Settlement
            2 = Port
            3 = Ruin
            4 = Forest
            5 = Mountain
        """
        if not 0 <= seed_index < 5:
            raise ValueError(f"seed_index must be 0-4, got {seed_index}")
        if not round_id:
            raise ValueError("round_id is required")
        
        # Convert numpy array to list if needed
        if hasattr(predictions, 'tolist'):
            predictions_list = predictions.tolist()
        else:
            predictions_list = predictions
        
        payload = {
            "round_id": round_id,
            "seed_index": seed_index,
            "prediction": predictions_list
        }
        
        response = self._make_request("/submit", method="POST", payload=payload)
        print(f"✓ Predictions submitted for seed {seed_index}")
        
        return response
    
    def get_budget(self) -> Dict[str, Any]:
        """
        Check your team's remaining query budget for the active round.
        
        Returns:
            Budget info:
            - round_id: UUID of active round
            - queries_used: Number of queries already used
            - queries_max: Total query budget (default 50)
            - active: Whether the round is still active
        
        Raises:
            requests.exceptions.RequestException: If request fails
        """
        response = self._make_request("/budget", method="GET")
        print(f"✓ Budget check: {response.get('queries_used', '?')}/{response.get('queries_max', '?')} queries used")
        return response
    
    def get_rounds(self) -> list:
        """
        List all rounds with status and timing.
        
        Returns:
            List of round objects with status, dates, map dimensions, etc.
        """
        response = self._make_request("/rounds", method="GET")
        return response
    
    def get_round_details(self, round_id: str) -> Dict[str, Any]:
        """
        Get detailed info for a specific round, including initial map states for all seeds.
        
        Args:
            round_id: UUID of the round
        
        Returns:
            Round details with initial terrain grids and settlement positions for all 5 seeds
        """
        response = self._make_request(f"/rounds/{round_id}", method="GET")
        return response
    
    def get_my_rounds(self) -> list:
        """
        Get all rounds enriched with your team's scores, budget, and rank.
        Requires team authentication.
        
        Returns:
            List of rounds with team-specific data (scores, submissions, budget, rank, etc.)
        """
        response = self._make_request("/my-rounds", method="GET")
        return response
    
    def get_my_predictions(self, round_id: str) -> list:
        """
        Get your team's submitted predictions for a round with argmax and confidence.
        Requires team authentication.
        
        Args:
            round_id: UUID of the round
        
        Returns:
            List of predictions with argmax grid, confidence grid, and score
        """
        response = self._make_request(f"/my-predictions/{round_id}", method="GET")
        return response
    
    def get_analysis(self, round_id: str, seed_index: int) -> Dict[str, Any]:
        """
        Get post-round analysis: your prediction vs ground truth for a seed.
        Only available after round is completed/scoring.
        
        Args:
            round_id: UUID of the round
            seed_index: Seed number (0-4)
        
        Returns:
            Analysis data with your prediction, ground truth, and score
        """
        response = self._make_request(f"/analysis/{round_id}/{seed_index}", method="GET")
        return response
    
    @property
    def api_call_count(self) -> int:
        """Return total number of API calls made."""
        return self.call_count
    
    @property
    def last_api_call(self) -> Optional[datetime]:
        """Return timestamp of last API call."""
        return self.last_call_time
