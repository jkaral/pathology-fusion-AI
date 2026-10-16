"""Pick the proposer backend from config, degrading gracefully.

``proposer: auto`` resolution order:
1. ``local_llm`` if a CUDA GPU and ``transformers`` are available (free, no key);
2. ``openai_compatible`` if the configured API-key environment variable is set;
3. ``deterministic`` otherwise (no key, no GPU, CI, tests).
An explicitly requested backend that cannot start raises, rather than silently changing.
"""

from __future__ import annotations

import importlib.util
import os

from pathfusion.agent.llm import LocalHFChat, OpenAICompatChat
from pathfusion.agent.proposers import DeterministicProposer, LLMProposer, Proposer
from pathfusion.agent.space import SearchSpace
from pathfusion.config import AgentCfg
from pathfusion.utils.logging import get_logger

log = get_logger("agent")


def resolve_backend(cfg: AgentCfg) -> str:
    if cfg.proposer != "auto":
        return cfg.proposer
    try:
        import torch

        has_gpu = torch.cuda.is_available()
    except ImportError:
        has_gpu = False
    if has_gpu and importlib.util.find_spec("transformers") is not None:
        return "local_llm"
    if os.environ.get(cfg.llm.api_key_env):
        return "openai_compatible"
    return "deterministic"


def make_proposer(cfg: AgentCfg, space: SearchSpace, n_dev: int, n_folds: int) -> Proposer:
    backend = resolve_backend(cfg)
    log.info("proposer backend: %s", backend)
    if backend == "deterministic":
        return DeterministicProposer(space)
    if backend == "local_llm":
        llm = LocalHFChat(cfg.llm)
    elif backend == "openai_compatible":
        llm = OpenAICompatChat(cfg.llm)
    else:
        raise ValueError(f"unknown proposer {backend!r}")
    return LLMProposer(space, llm, cfg.acceptance_z, cfg.llm.max_retries, n_dev=n_dev, n_folds=n_folds,
                       metric=cfg.selection_metric)
