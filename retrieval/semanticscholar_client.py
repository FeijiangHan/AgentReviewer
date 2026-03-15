import json
import urllib.parse
import urllib.request
from typing import Any, Dict, List


class SemanticScholarClient:
    BASE_URL = "https://api.semanticscholar.org/graph/v1"

    def __init__(self, api_key: str = "", timeout_s: float = 10.0):
        self.api_key = api_key
        self.timeout_s = timeout_s

    def _fetch_json(self, url: str) -> Dict[str, Any]:
        req = urllib.request.Request(url)
        if self.api_key:
            req.add_header("x-api-key", self.api_key)
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            return json.loads(resp.read().decode('utf-8'))

    def search_paper(self, query: str) -> List[Dict[str, Any]]:
        try:
            params = urllib.parse.urlencode({
                "query": query,
                "limit": 3,
                "fields": "title,authors,year,venue",
            })
            payload = self._fetch_json(f"{self.BASE_URL}/paper/search?{params}")
            return payload.get("data", [])
        except Exception:
            return []

    def search_author(self, name: str) -> List[Dict[str, Any]]:
        try:
            params = urllib.parse.urlencode({
                "query": name,
                "limit": 3,
                "fields": "name,affiliations,paperCount,hIndex",
            })
            payload = self._fetch_json(f"{self.BASE_URL}/author/search?{params}")
            return payload.get("data", [])
        except Exception:
            return []
