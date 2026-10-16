"""Chat backends for the LLM proposer. Both are optional; the agent degrades to the
deterministic proposer when neither is usable.

* ``LocalHFChat``: an open-weights instruct model run with ``transformers`` on the local
  GPU (default Qwen2.5-7B-Instruct, fp16 split over 2x T4). No API key, no cost.
* ``OpenAICompatChat``: any OpenAI-compatible ``/chat/completions`` endpoint (Gemini's
  free tier, Groq, a local vLLM/Ollama server...). The key is read from an environment
  variable named in the config and never written anywhere.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from abc import ABC, abstractmethod

from pathfusion.config import LLMCfg
from pathfusion.utils.logging import get_logger

log = get_logger("llm")


class ChatLLM(ABC):
    name: str = "llm"
    calls: int = 0

    @abstractmethod
    def chat(self, system: str, user: str) -> str: ...


class LocalHFChat(ChatLLM):
    def __init__(self, cfg: LLMCfg):
        self.cfg = cfg
        self.name = f"local:{cfg.local_model}"
        self._tok = None
        self._model = None
        self.calls = 0

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        t0 = time.perf_counter()
        self._tok = AutoTokenizer.from_pretrained(self.cfg.local_model)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(self.cfg.local_model, dtype=dtype, device_map="auto")
        log.info("loaded %s in %.0fs", self.cfg.local_model, time.perf_counter() - t0)

    def chat(self, system: str, user: str) -> str:
        import torch

        if self._model is None:
            self._load()
        assert self._tok is not None and self._model is not None
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        text = self._tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self._tok(text, return_tensors="pt").to(self._model.device)
        gen_kw: dict[str, object] = {"max_new_tokens": self.cfg.max_new_tokens}
        if self.cfg.temperature > 0:
            gen_kw.update(do_sample=True, temperature=self.cfg.temperature)
        else:
            gen_kw.update(do_sample=False)
        with torch.inference_mode():
            out = self._model.generate(**inputs, **gen_kw)
        self.calls += 1
        return self._tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


class OpenAICompatChat(ChatLLM):
    def __init__(self, cfg: LLMCfg):
        key = os.environ.get(cfg.api_key_env)
        if not key:
            raise RuntimeError(f"environment variable {cfg.api_key_env} is not set")
        self._key = key
        self.cfg = cfg
        self.name = f"api:{cfg.model}"
        self.calls = 0

    def chat(self, system: str, user: str) -> str:
        url = self.cfg.base_url.rstrip("/") + "/chat/completions"
        body = {
            "model": self.cfg.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_new_tokens,
        }
        req = urllib.request.Request(
            url, data=json.dumps(body).encode(), method="POST",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self._key}"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = json.loads(resp.read())
        self.calls += 1
        return payload["choices"][0]["message"]["content"]


def extract_json(text: str) -> dict:
    """First balanced ``{...}`` object in ``text`` (tolerates code fences and chatter)."""
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    raise ValueError("no JSON object found in LLM reply")
