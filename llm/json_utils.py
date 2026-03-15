import json
from typing import Any, Dict


def load_json_with_repair(text: str) -> Dict[str, Any]:
    """
    Best-effort JSON loader for LLM outputs.
    1) parse directly
    2) parse outermost {...} slice
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise
