"""Signal router + ensemble.

Combines signals from `POSITIVE_EDGE`-rated specialists, optionally filters
through the ML ranker, and applies the Risk Manager's veto before passing
signals to the engine for execution.

The ensemble is intentionally simple in its first revision:

* Specialists are loaded by module name from `src.strategy.specialists.*`.
* For each session day, each enabled specialist's `generate_signals(bars, params)`
  is called.
* The resulting signal list is sorted by `(timestamp, -confidence)`.
* If the ML ranker is enabled and trained, signals are re-scored
  (`metadata['ml_score']` populated); top-K per day are kept.
* The Risk Manager (if present) vetoes signals that violate hard rules.
* Finally signals are emitted in chronological order to the engine driver.
"""
from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import polars as pl

from src.common.types import Signal

log = logging.getLogger(__name__)


@dataclass
class EnsembleConfig:
    """Which specialists to include and which routing options to apply."""

    enabled_specialists: list[str] = field(default_factory=list)
    use_ml_ranker: bool = False
    ml_topk_per_day: Optional[int] = None  # keep top-K by ml_score per day
    use_risk_manager: bool = False
    specialist_params: dict[str, object] = field(default_factory=dict)


class Ensemble:
    def __init__(self, cfg: EnsembleConfig) -> None:
        self.cfg = cfg
        self._modules: dict[str, object] = {}
        for name in cfg.enabled_specialists:
            try:
                self._modules[name] = importlib.import_module(
                    f"src.strategy.specialists.{name}"
                )
            except ImportError as e:
                log.warning("specialist %s not importable: %s", name, e)

    def generate_for_session(self, sd: date, bars: pl.DataFrame) -> list[Signal]:
        all_sigs: list[Signal] = []
        for name, mod in self._modules.items():
            gen = getattr(mod, "generate_signals", None)
            if gen is None:
                continue
            params = self.cfg.specialist_params.get(name)
            if params is None:
                params_cls = next(
                    (
                        v for k, v in vars(mod).items()
                        if (
                            k.endswith("Params")
                            and isinstance(v, type)
                            and getattr(v, "__module__", "") == mod.__name__
                        )
                    ),
                    None,
                )
                if params_cls is not None:
                    try:
                        params = params_cls()
                    except TypeError:
                        params = None
            try:
                sigs = gen(bars, params) if params is not None else gen(bars)
            except Exception as e:
                log.warning("specialist %s failed on %s: %s", name, sd, e)
                continue
            all_sigs.extend(sigs)
        all_sigs.sort(key=lambda s: (s.timestamp, -s.confidence))

        if self.cfg.use_ml_ranker:
            try:
                ml_mod = importlib.import_module("src.strategy.specialists.ml_ranker")
                scorer = getattr(ml_mod, "score_signals", None)
                if scorer is not None:
                    all_sigs = scorer(all_sigs, bars, None)
            except Exception as e:
                log.warning("ml_ranker pass failed: %s", e)
            if self.cfg.ml_topk_per_day is not None:
                k = self.cfg.ml_topk_per_day
                all_sigs.sort(
                    key=lambda s: (-(s.metadata.get("ml_score", s.confidence)),),
                )
                all_sigs = all_sigs[:k]
                all_sigs.sort(key=lambda s: s.timestamp)
        return all_sigs

    def veto_for_account(self, account, signal, state: dict) -> tuple[bool, str]:
        if not self.cfg.use_risk_manager:
            return True, ""
        try:
            rm = importlib.import_module("src.strategy.specialists.risk_manager")
            should = getattr(rm, "should_take_signal", None)
            if should is None:
                return True, ""
            return should(account, signal, state)
        except Exception as e:
            log.warning("risk_manager veto failed: %s", e)
            return True, ""
