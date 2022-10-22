from dataclasses import dataclass
from typing import Any
from typing import Dict


@dataclass
class PreparedRequest:
    api_url: str
    headers: Dict[str, str]
    params: Dict[str, Any]


@dataclass
class Completion:
    prompt: str
    completion_text: str
    response: Dict[str, Any]
    params: Dict[str, Any]
