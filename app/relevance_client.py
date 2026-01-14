"""
Minimal RelevanceAI HTTP client (no SDK dependency).

Why:
- The official `relevanceai` Python SDK pins `requests==2.32.3`.
- Our LangChain stack needs `requests>=2.32.5`.
- Keeping both in one Render service causes dependency resolution conflicts.

We only implement the subset used by `app/main.py`:
- trigger_task
- schedule_action_in_task
- view_task_steps (raw JSON)
- get_task_output_preview (raw JSON)
- get_metadata
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests


@dataclass
class RelevanceClient:
    api_key: str
    region: str
    project: str
    base_url: str | None = None
    timeout_s: int = 60

    def __post_init__(self) -> None:
        if not self.base_url:
            self.base_url = f"https://api-{self.region}.stack.tryrelevance.com/latest"
        self.base_url = str(self.base_url).rstrip("/")

    @property
    def _headers(self) -> Dict[str, str]:
        # Matches SDK behavior: Authorization: "{project}:{api_key}"
        return {
            "Authorization": f"{self.project}:{self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _url(self, path: str) -> str:
        return f"{self.base_url}/{str(path).lstrip('/')}"

    def get_metadata(self, *, conversation_id: str) -> Dict[str, Any]:
        # Matches SDK: GET knowledge/sets/{conversation_id}/get_metadata
        r = requests.get(
            self._url(f"knowledge/sets/{conversation_id}/get_metadata"),
            headers=self._headers,
            timeout=self.timeout_s,
        )
        r.raise_for_status()
        data = r.json() if r.text else {}
        md = data.get("metadata")
        return md if isinstance(md, dict) else {}


@dataclass
class RelevanceAgent:
    client: RelevanceClient
    agent_id: str

    def trigger_task(self, *, message: str, **kwargs: Any) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "agent_id": self.agent_id,
            "message": {"role": "user", "content": message},
            **kwargs,
        }
        r = requests.post(
            self.client._url("agents/trigger"),
            headers=self.client._headers,
            json=body,
            timeout=self.client.timeout_s,
        )
        r.raise_for_status()
        return r.json() if r.text else {}

    def schedule_action_in_task(
        self,
        *,
        conversation_id: str,
        message: str,
        minutes_until_schedule: int = 0,
    ) -> Dict[str, Any]:
        body = {
            "conversation_id": conversation_id,
            "message": message,
            "minutes_until_schedule": int(minutes_until_schedule),
        }
        r = requests.post(
            self.client._url(f"agents/{self.agent_id}/scheduled_triggers_item/create"),
            headers=self.client._headers,
            json=body,
            timeout=self.client.timeout_s,
        )
        r.raise_for_status()
        return r.json() if r.text else {}

    def view_task_steps_raw(self, *, conversation_id: str) -> Dict[str, Any]:
        # Matches SDK: POST agents/{agent_id}/tasks/{conversation_id}/view
        r = requests.post(
            self.client._url(f"agents/{self.agent_id}/tasks/{conversation_id}/view"),
            headers=self.client._headers,
            timeout=self.client.timeout_s,
        )
        r.raise_for_status()
        return r.json() if r.text else {}

    def get_task_output_preview(self, *, conversation_id: str) -> Dict[str, Any]:
        # Matches SDK: GET agents/conversations/studios/list?conversation_id=...&agent_id=...&page_size=100
        params = {
            "conversation_id": conversation_id,
            "agent_id": self.agent_id,
            "page_size": 100,
        }
        r = requests.get(
            self.client._url("agents/conversations/studios/list"),
            headers=self.client._headers,
            params=params,
            timeout=self.client.timeout_s,
        )
        r.raise_for_status()
        return r.json() if r.text else {}

