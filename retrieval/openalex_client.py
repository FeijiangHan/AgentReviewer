import json
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional


class OpenAlexClient:
    BASE_WORKS = "https://api.openalex.org/works"
    BASE_AUTHORS = "https://api.openalex.org/authors"

    def __init__(self, timeout_s: float = 10.0):
        self.timeout_s = timeout_s

    def _fetch_json(self, url: str) -> Optional[Dict[str, Any]]:
        try:
            with urllib.request.urlopen(url, timeout=self.timeout_s) as resp:
                return json.loads(resp.read().decode('utf-8'))
        except Exception:
            return None

    def search_work_by_title(self, title: str) -> Optional[Dict[str, Any]]:
        params = urllib.parse.urlencode({"search": title, "per-page": 1})
        payload = self._fetch_json(f"{self.BASE_WORKS}?{params}")
        if not payload:
            return None
        results = payload.get("results", [])
        return results[0] if results else None

    def get_author(self, author_id: str) -> Optional[Dict[str, Any]]:
        return self._fetch_json(f"{self.BASE_AUTHORS}/{author_id}")

    def search_author_by_name(self, name: str) -> List[Dict[str, Any]]:
        params = urllib.parse.urlencode({"search": name, "per-page": 3})
        payload = self._fetch_json(f"{self.BASE_AUTHORS}?{params}")
        return payload.get("results", []) if payload else []
