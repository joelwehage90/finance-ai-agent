"""
Minimal Supabase (PostgREST) client.

Why:
- The official `supabase` Python SDK currently pulls in dependencies that conflict
  with `relevanceai==10.2.2` (pydantic pinning).
- This project only needs a subset of PostgREST functionality:
  table().select/insert/update + filters (eq/in_/gte/lt) + order/limit/range + rpc().

This client is intentionally small, explicit, and compatible with our existing call sites.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import requests

from .settings import settings


@dataclass
class SupabaseResponse:
    data: Any
    error: Optional[Dict[str, Any]] = None
    status_code: Optional[int] = None
    raw: Optional[Any] = None


def _fmt_scalar(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def _fmt_in_list(values: List[Any]) -> str:
    # PostgREST format: in.(a,b) ; strings should be quoted.
    out: List[str] = []
    for v in values:
        if v is None:
            out.append("null")
        elif isinstance(v, str):
            s = v.replace('"', '\\"')
            out.append(f'"{s}"')
        elif isinstance(v, bool):
            out.append("true" if v else "false")
        else:
            out.append(str(v))
    return ",".join(out)


class _PostgrestQuery:
    def __init__(self, *, base_url: str, headers: Dict[str, str]):
        self._base_url = base_url.rstrip("/")
        self._headers = dict(headers)
        self._method: str = "GET"
        self._select: Optional[str] = None
        self._payload: Any = None
        self._filters: List[Tuple[str, str]] = []
        self._orders: List[str] = []
        self._limit: Optional[int] = None
        self._offset: Optional[int] = None

    # builder methods
    def select(self, columns: str) -> "_PostgrestQuery":
        self._method = "GET"
        self._select = columns
        return self

    def insert(self, payload: Any) -> "_PostgrestQuery":
        self._method = "POST"
        self._payload = payload
        return self

    def update(self, payload: Any) -> "_PostgrestQuery":
        self._method = "PATCH"
        self._payload = payload
        return self

    def eq(self, col: str, val: Any) -> "_PostgrestQuery":
        self._filters.append((col, f"eq.{_fmt_scalar(val)}"))
        return self

    def gte(self, col: str, val: Any) -> "_PostgrestQuery":
        self._filters.append((col, f"gte.{_fmt_scalar(val)}"))
        return self

    def lt(self, col: str, val: Any) -> "_PostgrestQuery":
        self._filters.append((col, f"lt.{_fmt_scalar(val)}"))
        return self

    def in_(self, col: str, values: List[Any]) -> "_PostgrestQuery":
        self._filters.append((col, f"in.({_fmt_in_list(values)})"))
        return self

    def or_(self, expr: str) -> "_PostgrestQuery":
        """
        Minimal PostgREST OR support.

        PostgREST syntax: ?or=(a.eq.1,b.eq.2)
        Callers typically pass: "a.eq.1,b.eq.2" (without parentheses).

        Notes:
        - This client stores query params in a dict on execute(), so multiple `or_()` calls
          will overwrite each other. Keep usage to one per query (matches our tool patterns).
        """
        e = str(expr or "").strip()
        if not e:
            return self
        if not (e.startswith("(") and e.endswith(")")):
            e = f"({e})"
        self._filters.append(("or", e))
        return self

    def order(self, col: str, desc: bool = False) -> "_PostgrestQuery":
        self._orders.append(f"{col}.{'desc' if desc else 'asc'}")
        return self

    def limit(self, n: int) -> "_PostgrestQuery":
        self._limit = int(n)
        return self

    def range(self, start: int, end: int) -> "_PostgrestQuery":
        # Translate PostgREST range to offset/limit for simplicity.
        start_i = int(start)
        end_i = int(end)
        if end_i < start_i:
            self._offset = start_i
            self._limit = 0
        else:
            self._offset = start_i
            self._limit = end_i - start_i + 1
        return self

    def execute(self) -> SupabaseResponse:
        params: Dict[str, str] = {}
        if self._select is not None:
            params["select"] = self._select
        if self._orders:
            params["order"] = ",".join(self._orders)
        if self._limit is not None:
            params["limit"] = str(self._limit)
        if self._offset is not None:
            params["offset"] = str(self._offset)
        for k, v in self._filters:
            params[k] = v

        headers = dict(self._headers)
        # For write ops, ask PostgREST to return rows.
        if self._method in {"POST", "PATCH"}:
            headers.setdefault("Prefer", "return=representation")

        # Cloudflare/Supabase can occasionally return transient 5xx/522s.
        # Keep retries small and fast to avoid blowing up request latency.
        retry_delays_s = [0.6, 1.2]
        last_exc: Optional[Exception] = None
        resp = None

        for attempt in range(len(retry_delays_s) + 1):
            try:
                resp = requests.request(
                    self._method,
                    self._base_url,
                    params=params,
                    json=self._payload if self._method in {"POST", "PATCH"} else None,
                    headers=headers,
                    timeout=60,
                )
                # Retry on transient infra errors
                if resp.status_code in {502, 503, 504, 522}:
                    if attempt < len(retry_delays_s):
                        time.sleep(retry_delays_s[attempt])
                        continue
                break
            except Exception as e:
                last_exc = e
                if attempt < len(retry_delays_s):
                    time.sleep(retry_delays_s[attempt])
                    continue
                return SupabaseResponse(data=None, error={"message": str(e), "type": type(e).__name__})

        if resp is None:
            return SupabaseResponse(data=None, error={"message": str(last_exc), "type": type(last_exc).__name__ if last_exc else "UnknownError"})

        try:
            data = resp.json() if resp.text else None
        except Exception:
            data = resp.text

        if resp.status_code >= 400:
            return SupabaseResponse(
                data=None,
                error={"status_code": resp.status_code, "body": data},
                status_code=resp.status_code,
                raw=resp,
            )

        return SupabaseResponse(data=data, error=None, status_code=resp.status_code, raw=resp)


class SupabaseClient:
    def __init__(self, url: str, service_key: str):
        self._url = url.rstrip("/")
        self._service_key = service_key
        self._headers = {
            "apikey": service_key,
            "Authorization": f"Bearer {service_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def table(self, name: str) -> _PostgrestQuery:
        # PostgREST endpoint
        base_url = f"{self._url}/rest/v1/{quote(name)}"
        return _PostgrestQuery(base_url=base_url, headers=self._headers)

    def rpc(self, fn_name: str, args: Dict[str, Any]) -> _PostgrestQuery:
        base_url = f"{self._url}/rest/v1/rpc/{quote(fn_name)}"
        q = _PostgrestQuery(base_url=base_url, headers=self._headers)
        q._method = "POST"
        q._payload = args
        return q


supabase = SupabaseClient(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)
