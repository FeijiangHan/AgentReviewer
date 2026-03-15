import json
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional


class CrossrefClient:
    BASE_URL = "https://api.crossref.org/works"

    def __init__(self, timeout_s: float = 10.0):
        self.timeout_s = timeout_s

    def search_work_by_title(self, title: str) -> Optional[Dict[str, Any]]:
        try:
            params = urllib.parse.urlencode({"query.title": title, "rows": 1})
            with urllib.request.urlopen(f"{self.BASE_URL}?{params}", timeout=self.timeout_s) as resp:
                payload = json.loads(resp.read().decode('utf-8'))
            items = payload.get("message", {}).get("items", [])
            return items[0] if items else None
        except Exception:
            return None
