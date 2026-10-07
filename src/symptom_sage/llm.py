"""LLM adapters behind one interface: an OpenAI-compatible HTTP client and offline fakes."""
from __future__ import annotations

import re
from typing import Protocol

import httpx


class LLMError(RuntimeError):
    pass


class LLM(Protocol):
    name: str

    def complete(self, system: str, user: str, *, json_mode: bool = False) -> str: ...


class OpenAIChatLLM:
    """``POST {base_url}/chat/completions`` with temperature 0 and a fixed seed."""

    def __init__(self, model: str, api_key: str, base_url: str = "https://api.openai.com/v1", seed: int = 7,
                 timeout_s: float = 60.0, client: httpx.Client | None = None):
        if not api_key:
            raise LLMError("OPENAI_API_KEY is not set; use SYMPTOM_SAGE_LLM=fake for offline runs")
        self.name = f"openai:{model}"
        self._model, self._seed = model, seed
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._client = client or httpx.Client(timeout=timeout_s)

    def complete(self, system: str, user: str, *, json_mode: bool = False) -> str:
        body = {"model": self._model, "temperature": 0, "seed": self._seed,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        try:
            r = self._client.post(self._url, json=body, headers=self._headers)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"] or ""
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            raise LLMError(f"{self.name}: {type(exc).__name__}: {exc}") from exc


class ScriptedLLM:
    """Returns prepared replies in order. For tests."""

    name = "scripted"

    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.prompts: list[tuple[str, str]] = []

    def complete(self, system: str, user: str, *, json_mode: bool = False) -> str:
        self.prompts.append((system, user))
        if not self.replies:
            raise LLMError("no scripted replies left")
        return self.replies.pop(0)


_CASE_LINE = re.compile(r"^\[(?P<id>[A-Z]{3}\d{4})\] .*?Assessment: (?P<cond>[^.|]+)", re.MULTILINE)


class TemplateLLM:
    """Offline answer writer. It reads the cases in the prompt and lists their assessments.

    It writes only what the context holds, with a citation for each condition. It shows the
    answer format and lets the demo run with no key. It does not reason.
    """

    name = "template"

    def complete(self, system: str, user: str, *, json_mode: bool = False) -> str:
        seen: dict[str, list[str]] = {}
        for m in _CASE_LINE.finditer(user):
            seen.setdefault(m.group("cond").strip(), []).append(m.group("id"))
        if not seen:
            return "The similar cases do not give a clear match. Please talk with a doctor."
        lines = ["Conditions that similar cases were assessed with:"]
        for cond, ids in list(seen.items())[:3]:
            lines.append(f"- {cond} " + " ".join(f"[{i}]" for i in ids))
        lines.append("Only a doctor who examines you can tell which, if any, applies to you.")
        return "\n".join(lines)
