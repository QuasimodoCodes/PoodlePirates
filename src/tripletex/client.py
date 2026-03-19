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
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential


def _is_retryable(exc: BaseException) -> bool:
    """Only retry on network errors, 429, and 5xx. Never retry 4xx client errors."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    return isinstance(exc, (httpx.TimeoutException, httpx.NetworkError))


class TripletexClient:
    """
    Thin httpx wrapper that handles:
    - HTTP Basic Auth  (username="0", password=<session_token>)
    - Base URL from the per-request proxy URL
    - Retry on 429 and 5xx only (never on 4xx client errors)
    """

    def __init__(self, proxy_url: str, session_token: str) -> None:
        self.base_url = proxy_url.rstrip("/")
        self._auth = ("0", session_token)
        self._headers = {"Accept": "application/json", "Content-Type": "application/json"}

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    @retry(retry=retry_if_exception(_is_retryable), stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    def get(self, path: str, params: dict | None = None) -> dict:
        resp = httpx.get(self._url(path), auth=self._auth, headers=self._headers, params=params, timeout=20)
        resp.raise_for_status()
        return resp.json()

    @retry(retry=retry_if_exception(_is_retryable), stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    def post(self, path: str, body: dict) -> dict:
        resp = httpx.post(self._url(path), auth=self._auth, headers=self._headers, json=body, timeout=20)
        resp.raise_for_status()
        return resp.json()

    @retry(retry=retry_if_exception(_is_retryable), stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    def put(self, path: str, body: dict, params: dict | None = None) -> dict:
        resp = httpx.put(
            self._url(path), auth=self._auth, headers=self._headers, json=body, params=params, timeout=20
        )
        resp.raise_for_status()
        return resp.json()

    @retry(retry=retry_if_exception(_is_retryable), stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    def delete(self, path: str) -> None:
        resp = httpx.delete(self._url(path), auth=self._auth, headers=self._headers, timeout=20)
        resp.raise_for_status()
