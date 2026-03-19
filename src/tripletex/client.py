"""
Tripletex HTTP client — per-request credentials.

The competition platform sends a proxy_url and session_token with each
/solve request. We never read from env vars here; credentials are
injected at construction time.

Usage:
    client = TripletexClient(proxy_url="https://...", session_token="abc123")
    company = client.get("/company")
"""
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential


class TripletexClient:
    """
    Thin httpx wrapper that handles:
    - HTTP Basic Auth  (username="0", password=<session_token>)
    - Base URL from the per-request proxy URL
    - Automatic retry on transient errors (429, 5xx)
    """

    def __init__(self, proxy_url: str, session_token: str) -> None:
        # Strip trailing slash so we can always do base_url + "/path"
        self.base_url = proxy_url.rstrip("/")
        self._auth = ("0", session_token)
        self._headers = {"Accept": "application/json", "Content-Type": "application/json"}

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    def get(self, path: str, params: dict | None = None) -> dict:
        resp = httpx.get(self._url(path), auth=self._auth, headers=self._headers, params=params, timeout=20)
        resp.raise_for_status()
        return resp.json()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    def post(self, path: str, body: dict) -> dict:
        resp = httpx.post(self._url(path), auth=self._auth, headers=self._headers, json=body, timeout=20)
        resp.raise_for_status()
        return resp.json()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    def put(self, path: str, body: dict, params: dict | None = None) -> dict:
        resp = httpx.put(
            self._url(path), auth=self._auth, headers=self._headers, json=body, params=params, timeout=20
        )
        resp.raise_for_status()
        return resp.json()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    def delete(self, path: str) -> None:
        resp = httpx.delete(self._url(path), auth=self._auth, headers=self._headers, timeout=20)
        resp.raise_for_status()
