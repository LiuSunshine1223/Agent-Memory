"""Small Ollama client used by the local workflow."""

import requests
from urllib.parse import urlsplit, urlunsplit


class LLMUnavailableError(RuntimeError):
    """Raised when Ollama cannot return a valid answer."""


class LLMModel:
    def __init__(self, url, model_name, timeout=120, *, options=None):
        self.url = url
        self.model = model_name
        self.timeout = float(timeout)
        self.options = {"temperature": 0.0, "top_p": 0.1, **dict(options or {})}
        self.last_response_metadata = {}

    def ask(self, prompt, *, system=None):
        return self._generate(prompt, system=system)

    def ask_json(self, prompt, *, schema=None, system=None):
        """Constrain syntax for structured tasks; callers still validate schema."""
        return self._generate(prompt, json_mode=True, schema=schema, system=system)

    def chat(self, messages):
        """Preserve real user/assistant turns through Ollama's chat endpoint."""
        self.last_response_metadata = {}
        parts = urlsplit(self.url)
        endpoint = urlunsplit(parts._replace(path=parts.path.rsplit("/", 1)[0] + "/chat"))
        try:
            response = requests.post(endpoint, json={
                "model": self.model, "messages": messages, "stream": False,
                "options": self.options,
            }, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
            self.last_response_metadata = {key: payload.get(key) for key in (
                "model", "done", "done_reason", "prompt_eval_count", "eval_count",
                "total_duration", "load_duration", "prompt_eval_duration", "eval_duration",
            )}
            answer = payload.get("message", {}).get("content")
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("Ollama chat returned no answer")
            return answer.strip()
        except (requests.RequestException, ValueError, AttributeError) as exc:
            raise LLMUnavailableError(f"Ollama chat failed: {exc}") from exc

    def _generate(self, prompt, *, json_mode=False, system=None, schema=None):
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": self.options,
        }
        if json_mode:
            payload["format"] = schema or "json"
        if system is not None:
            payload["system"] = system
        try:
            response = requests.post(
                self.url,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            answer = response.json().get("response")
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("Ollama response has no non-empty response field")
            return answer.strip()
        except (requests.RequestException, ValueError) as exc:
            raise LLMUnavailableError(f"Ollama request failed: {exc}") from exc
