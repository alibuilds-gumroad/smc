"""
SMC Engine V2.1
===============

Central causal orchestrator for the SMC stack.

Pipeline
--------

Market Data
    |
    +--> Structure
    |
    +--> Liquidity
    |
    +--> FVG
    |
    +--> Order Block
    |
    +--> Zones
    |
    +--> Context
    |
    +--> Directional Score
    |
    +--> Signal

Important
---------
This module does NOT independently detect SMC patterns.

It orchestrates the dedicated modules:

    structure.py
    liquidity.py
    fvg.py
    order_block.py
    zones.py

Anti-lookahead rule
-------------------
For candle i, ONLY:

    df.iloc[:i + 1]

is allowed.

Every object must additionally expose:

    confirmation_index

and:

    confirmation_index <= current_index

to be considered known.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
from datetime import datetime
import math

import pandas as pd
import numpy as np


# ============================================================
# CONSTANTS
# ============================================================

DEFAULT_LOOKBACK = 150
DEFAULT_MIN_SIGNAL_SCORE = 50.0
DEFAULT_MIN_DIRECTION_EDGE = 10.0

DEFAULT_CLOSED_CANDLE_ONLY = True

# Maximum contribution of each context.
MAX_STRUCTURE_SCORE = 25.0
MAX_BOS_SCORE = 15.0
MAX_MSS_SCORE = 20.0
MAX_CHOCH_SCORE = 25.0

MAX_LIQUIDITY_SCORE = 20.0
MAX_ZONE_SCORE = 20.0


# ============================================================
# HELPERS
# ============================================================

def _get(
    obj: Any,
    name: str,
    default: Any = None,
) -> Any:

    if obj is None:
        return default

    if isinstance(obj, dict):
        return obj.get(
            name,
            default,
        )

    return getattr(
        obj,
        name,
        default,
    )


def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:

    try:

        result = float(value)

        if math.isfinite(result):
            return result

    except Exception:
        pass

    return default


def _safe_int(
    value: Any,
    default: int = -1,
) -> int:

    try:
        return int(value)
    except Exception:
        return default


def _timestamp(
    value: Any,
) -> Optional[datetime]:

    if value is None:
        return None

    if isinstance(value, datetime):
        return value

    try:

        value = pd.Timestamp(value)

        if pd.isna(value):
            return None

        return value.to_pydatetime()

    except Exception:
        return None


def _normalize_direction(
    value: Any,
) -> Optional[str]:

    if value is None:
        return None

    value = str(
        value
    ).lower().strip()

    mapping = {
        "bullish": "bullish",
        "buy": "bullish",
        "long": "bullish",
        "up": "bullish",

        "bearish": "bearish",
        "sell": "bearish",
        "short": "bearish",
        "down": "bearish",
    }

    return mapping.get(
        value,
        value,
    )


def _event_type(
    event: Any,
) -> str:

    value = _get(
        event,
        "event_type",
        None,
    )

    if value is None:
        value = _get(
            event,
            "type",
            "",
        )

    return str(
        value or ""
    ).upper()


# ============================================================
# SMC EVENT
# ============================================================

@dataclass
class SMCEvent:
    """
    Normalized event emitted by the orchestrator.

    This is NOT a replacement for the underlying SMC objects.
    """

    event_type: str

    direction: Optional[str]

    index: int

    confirmation_index: int

    timestamp: Optional[datetime]

    source: str

    source_id: Optional[str] = None

    price: Optional[float] = None

    score: float = 0.0

    details: Dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# CONTEXT
# ============================================================

@dataclass
class SMCContext:

    current_index: int

    current_timestamp: Optional[datetime]

    structure_bias: Optional[str] = None

    structure_state: Optional[str] = None

    latest_structure_event: Any = None

    latest_bos: Any = None
    latest_mss: Any = None
    latest_choch: Any = None

    latest_liquidity_sweep: Any = None

    best_zone: Any = None

    bullish_structure_score: float = 0.0
    bearish_structure_score: float = 0.0

    bullish_liquidity_score: float = 0.0
    bearish_liquidity_score: float = 0.0

    bullish_zone_score: float = 0.0
    bearish_zone_score: float = 0.0

    bullish_total_score: float = 0.0
    bearish_total_score: float = 0.0

    signal: str = "neutral"

    confidence: float = 0.0

    reasons: List[str] = field(
        default_factory=list
    )


# ============================================================
# RESULT
# ============================================================

@dataclass
class SMCResult:

    success: bool = False

    current_index: int = -1

    current_timestamp: Optional[datetime] = None

    # --------------------------------------------------------
    # Module results
    # --------------------------------------------------------

    structure: Any = None

    liquidity: Any = None

    fvg: Any = None

    order_blocks: Any = None

    zones: Any = None

    # --------------------------------------------------------
    # Context
    # --------------------------------------------------------

    context: Optional[SMCContext] = None

    # --------------------------------------------------------
    # Signal
    # --------------------------------------------------------

    signal: str = "neutral"

    direction: Optional[str] = None

    score: float = 0.0

    bullish_score: float = 0.0

    bearish_score: float = 0.0

    confidence: float = 0.0

    # --------------------------------------------------------
    # Events
    # --------------------------------------------------------

    events: List[SMCEvent] = field(
        default_factory=list
    )

    # --------------------------------------------------------
    # Errors
    # --------------------------------------------------------

    module_errors: Dict[str, str] = field(
        default_factory=dict
    )

    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:

        return asdict(self)


# ============================================================
# ENGINE
# ============================================================

class SMCEngine:
    """
    Causal SMC orchestrator V2.1.
    """

    def __init__(
        self,
        lookback: int = DEFAULT_LOOKBACK,
        min_signal_score: float = DEFAULT_MIN_SIGNAL_SCORE,
        min_direction_edge: float = DEFAULT_MIN_DIRECTION_EDGE,
        closed_candle_only: bool = DEFAULT_CLOSED_CANDLE_ONLY,
    ):

        self.lookback = max(
            20,
            int(lookback),
        )

        self.min_signal_score = float(
            min_signal_score
        )

        self.min_direction_edge = float(
            min_direction_edge
        )

        self.closed_candle_only = bool(
            closed_candle_only
        )

    # ========================================================
    # PUBLIC ANALYZE
    # ========================================================

    def analyze(
        self,
        df: pd.DataFrame,
        current_index: Optional[int] = None,

        structure_result: Any = None,
        liquidity_result: Any = None,
        fvg_result: Any = None,
        order_block_result: Any = None,
        zone_result: Any = None,

        allow_incomplete_candle: bool = False,
    ) -> SMCResult:

        try:

            data = self._validate_dataframe(
                df
            )

            if data.empty:

                return SMCResult(
                    success=False,
                    error="Empty dataframe",
                )

            # ------------------------------------------------
            # Current candle
            # ------------------------------------------------

            if current_index is None:

                current_index = (
                    len(data) - 2
                    if (
                        self.closed_candle_only
                        and not allow_incomplete_candle
                        and len(data) >= 2
                    )
                    else len(data) - 1
                )

            current_index = int(
                current_index
            )

            if current_index < 0:

                return SMCResult(
                    success=False,
                    error="current_index < 0",
                )

            if current_index >= len(data):

                current_index = (
                    len(data) - 1
                )

            # ------------------------------------------------
            # CRITICAL CAUSAL SNAPSHOT
            #
            # Nothing after current_index is visible.
            # ------------------------------------------------

            snapshot = data.iloc[
                : current_index + 1
            ].copy()

            # ------------------------------------------------
            # Apply lookback WITHOUT destroying the global
            # positional index.
            #
            # We calculate a start index, then pass a slice
            # whose rows keep the original positional index.
            #
            # Upstream modules in V2.1 should treat positional
            # indexes consistently.
            # ------------------------------------------------

            start_index = max(
                0,
                current_index
                - self.lookback
                + 1,
            )

            work = snapshot.iloc[
                start_index:
            ].copy()

            # ------------------------------------------------
            # Local index inside module
            #
            # IMPORTANT:
            # We preserve original candle index in a dedicated
            # column so module confirmation indexes can be
            # mapped back safely.
            # ------------------------------------------------

            work["_engine_index"] = np.arange(
                start_index,
                current_index + 1,
            )

            current_timestamp = (
                self._row_timestamp(
                    snapshot.iloc[-1]
                )
            )

            # =================================================
            # MODULES
            # =================================================

            if structure_result is None:

                structure_result = (
                    self._run_structure(
                        work,
                        current_index=current_index,
                    )
                )

            if liquidity_result is None:

                liquidity_result = (
                    self._run_liquidity(
                        work,
                        current_index=current_index,
                        structure_result=structure_result,
                    )
                )

            if fvg_result is None:

                fvg_result = (
                    self._run_fvg(
                        work,
                        current_index=current_index,
                    )
                )

            if order_block_result is None:

                order_block_result = (
                    self._run_order_blocks(
                        work,
                        current_index=current_index,
                        structure_result=structure_result,
                        liquidity_result=liquidity_result,
                    )
                )

            if zone_result is None:

                zone_result = (
                    self._run_zones(
                        work,
                        current_index=current_index,
                        structure_result=structure_result,
                        liquidity_result=liquidity_result,
                        fvg_result=fvg_result,
                        order_block_result=order_block_result,
                    )
                )

            # =================================================
            # CONTEXT
            # =================================================

            context = self._build_context(
                current_index=current_index,
                current_timestamp=current_timestamp,
                structure_result=structure_result,
                liquidity_result=liquidity_result,
                zone_result=zone_result,
            )

            # =================================================
            # EVENTS
            # =================================================

            events = self._collect_events(
                current_index=current_index,
                structure_result=structure_result,
                liquidity_result=liquidity_result,
                fvg_result=fvg_result,
                order_block_result=order_block_result,
                zone_result=zone_result,
            )

            # =================================================
            # RESULT
            # =================================================

            module_errors = {}

            for name, result in (
                ("structure", structure_result),
                ("liquidity", liquidity_result),
                ("fvg", fvg_result),
                ("order_blocks", order_block_result),
                ("zones", zone_result),
            ):

                error = _get(
                    result,
                    "error",
                    None,
                )

                if error:
                    module_errors[name] = str(
                        error
                    )

            return SMCResult(
                success=True,

                current_index=current_index,

                current_timestamp=current_timestamp,

                structure=structure_result,

                liquidity=liquidity_result,

                fvg=fvg_result,

                order_blocks=order_block_result,

                zones=zone_result,

                context=context,

                signal=context.signal,

                direction=(
                    context.signal
                    if context.signal
                    in (
                        "bullish",
                        "bearish",
                    )
                    else None
                ),

                score=max(
                    context.bullish_total_score,
                    context.bearish_total_score,
                ),

                bullish_score=(
                    context.bullish_total_score
                ),

                bearish_score=(
                    context.bearish_total_score
                ),

                confidence=context.confidence,

                events=events,

                module_errors=module_errors,
            )

        except Exception as exc:

            return SMCResult(
                success=False,
                current_index=(
                    int(current_index)
                    if current_index is not None
                    else -1
                ),
                error=(
                    f"{type(exc).__name__}: {exc}"
                ),
            )

    # ========================================================
    # MODULE RUNNERS
    # ========================================================

    def _run_structure(
        self,
        df: pd.DataFrame,
        current_index: int,
    ):

        from .structure import (
            MarketStructureEngine
        )

        engine = MarketStructureEngine()

        return engine.analyze(
            df,
            current_index=current_index,
        )

    def _run_liquidity(
        self,
        df: pd.DataFrame,
        current_index: int,
        structure_result: Any = None,
    ):

        from .liquidity import (
            LiquidityEngine
        )

        engine = LiquidityEngine()

        return engine.analyze(
            df,
            current_index=current_index,
            structure_result=structure_result,
        )

    def _run_fvg(
        self,
        df: pd.DataFrame,
        current_index: int,
    ):

        from .fvg import (
            FVGEngine
        )

        engine = FVGEngine(
            lookback=self.lookback,
        )

        return engine.analyze(
            df,
            current_index=current_index,
        )

    def _run_order_blocks(
        self,
        df: pd.DataFrame,
        current_index: int,
        structure_result: Any = None,
        liquidity_result: Any = None,
    ):

        from .order_block import (
            OrderBlockEngine
        )

        engine = OrderBlockEngine(
            lookback=self.lookback,
        )

        return engine.analyze(
            df,
            current_index=current_index,
            structure_result=structure_result,
            liquidity_result=liquidity_result,
        )

    def _run_zones(
        self,
        df: pd.DataFrame,
        current_index: int,
        structure_result: Any = None,
        liquidity_result: Any = None,
        fvg_result: Any = None,
        order_block_result: Any = None,
    ):

        from .zones import (
            ZoneEngine
        )

        engine = ZoneEngine(
            lookback=self.lookback,
        )

        return engine.analyze(
            df,
            current_index=current_index,
            structure_result=structure_result,
            liquidity_result=liquidity_result,
            fvg_result=fvg_result,
            order_block_result=order_block_result,
        )

    # ========================================================
    # CONTEXT
    # ========================================================

    def _build_context(
        self,
        current_index: int,
        current_timestamp: Optional[datetime],
        structure_result: Any,
        liquidity_result: Any,
        zone_result: Any,
    ) -> SMCContext:

        context = SMCContext(
            current_index=current_index,
            current_timestamp=current_timestamp,
        )

        # ----------------------------------------------------
        # Structure
        # ----------------------------------------------------

        structure_bias = _get(
            structure_result,
            "trend_bias",
            None,
        )

        context.structure_bias = (
            _normalize_direction(
                structure_bias
            )
        )

        context.structure_state = _get(
            structure_result,
            "structure_state",
            None,
        )

        events = self._confirmed_events(
            structure_result,
            current_index,
        )

        if events:

            context.latest_structure_event = max(
                events,
                key=lambda e: _safe_int(
                    _get(
                        e,
                        "confirmation_index",
                    )
                ),
            )

            context.latest_bos = self._latest_event(
                events,
                "BOS",
            )

            context.latest_mss = self._latest_event(
                events,
                "MSS",
            )

            context.latest_choch = self._latest_event(
                events,
                "CHOCH",
            )

        # ----------------------------------------------------
        # Structure directional score
        # ----------------------------------------------------

        self._score_structure(
            context,
            current_index,
        )

        # ----------------------------------------------------
        # Liquidity
        # ----------------------------------------------------

        sweep = self._latest_confirmed_sweep(
            liquidity_result,
            current_index,
        )

        context.latest_liquidity_sweep = sweep

        if sweep is not None:

            reaction = _normalize_direction(
                _get(
                    sweep,
                    "reaction",
                    _get(
                        sweep,
                        "direction",
                    ),
                )
            )

            strength = _safe_float(
                _get(
                    sweep,
                    "strength",
                    0.0,
                )
            )

            liquidity_score = min(
                MAX_LIQUIDITY_SCORE,
                10.0
                + max(
                    0.0,
                    strength - 1.0,
                ) * 2.0,
            )

            if reaction == "bullish":

                context.bullish_liquidity_score = (
                    liquidity_score
                )

                context.reasons.append(
                    "Bullish liquidity sweep"
                )

            elif reaction == "bearish":

                context.bearish_liquidity_score = (
                    liquidity_score
                )

                context.reasons.append(
                    "Bearish liquidity sweep"
                )

        # ----------------------------------------------------
        # Zone
        # ----------------------------------------------------

        best_zone = self._best_confirmed_zone(
            zone_result,
            current_index,
        )

        context.best_zone = best_zone

        if best_zone is not None:

            zone_direction = _normalize_direction(
                _get(
                    best_zone,
                    "direction",
                )
            )

            zone_score = min(
                MAX_ZONE_SCORE,
                max(
                    0.0,
                    _safe_float(
                        _get(
                            best_zone,
                            "score",
                            0.0,
                        )
                    )
                    * 0.20,
                ),
            )

            if zone_direction == "bullish":

                context.bullish_zone_score = (
                    zone_score
                )

                context.reasons.append(
                    "Bullish SMC zone"
                )

            elif zone_direction == "bearish":

                context.bearish_zone_score = (
                    zone_score
                )

                context.reasons.append(
                    "Bearish SMC zone"
                )

        # ----------------------------------------------------
        # Total
        # ----------------------------------------------------

        context.bullish_total_score = round(
            min(
                100.0,
                context.bullish_structure_score
                + context.bullish_liquidity_score
                + context.bullish_zone_score,
            ),
            2,
        )

        context.bearish_total_score = round(
            min(
                100.0,
                context.bearish_structure_score
                + context.bearish_liquidity_score
                + context.bearish_zone_score,
            ),
            2,
        )

        # ----------------------------------------------------
        # Signal
        # ----------------------------------------------------

        (
            context.signal,
            context.confidence,
        ) = self._resolve_signal(
            context.bullish_total_score,
            context.bearish_total_score,
        )

        return context

    # ========================================================
    # STRUCTURE SCORE
    # ========================================================

    def _score_structure(
        self,
        context: SMCContext,
        current_index: int,
    ):

        bias = context.structure_bias

        if bias == "bullish":

            context.bullish_structure_score += (
                MAX_STRUCTURE_SCORE
            )

        elif bias == "bearish":

            context.bearish_structure_score += (
                MAX_STRUCTURE_SCORE
            )

        # ----------------------------------------------------
        # BOS
        # ----------------------------------------------------

        bos = context.latest_bos

        if bos is not None:

            if self._is_known_at(
                bos,
                current_index,
            ):

                direction = _normalize_direction(
                    _get(
                        bos,
                        "direction",
                    )
                )

                if direction == "bullish":

                    context.bullish_structure_score += (
                        MAX_BOS_SCORE
                    )

                elif direction == "bearish":

                    context.bearish_structure_score += (
                        MAX_BOS_SCORE
                    )

        # ----------------------------------------------------
        # MSS
        # ----------------------------------------------------

        mss = context.latest_mss

        if mss is not None:

            if self._is_known_at(
                mss,
                current_index,
            ):

                direction = _normalize_direction(
                    _get(
                        mss,
                        "direction",
                    )
                )

                if direction == "bullish":

                    context.bullish_structure_score += (
                        MAX_MSS_SCORE
                    )

                elif direction == "bearish":

                    context.bearish_structure_score += (
                        MAX_MSS_SCORE
                    )

        # ----------------------------------------------------
        # CHOCH
        # ----------------------------------------------------

        choch = context.latest_choch

        if choch is not None:

            if self._is_known_at(
                choch,
                current_index,
            ):

                direction = _normalize_direction(
                    _get(
                        choch,
                        "direction",
                    )
                )

                if direction == "bullish":

                    context.bullish_structure_score += (
                        MAX_CHOCH_SCORE
                    )

                elif direction == "bearish":

                    context.bearish_structure_score += (
                        MAX_CHOCH_SCORE
                    )

        context.bullish_structure_score = min(
            MAX_STRUCTURE_SCORE
            + MAX_BOS_SCORE
            + MAX_MSS_SCORE
            + MAX_CHOCH_SCORE,
            context.bullish_structure_score,
        )

        context.bearish_structure_score = min(
            MAX_STRUCTURE_SCORE
            + MAX_BOS_SCORE
            + MAX_MSS_SCORE
            + MAX_CHOCH_SCORE,
            context.bearish_structure_score,
        )

    # ========================================================
    # SIGNAL
    # ========================================================

    def _resolve_signal(
        self,
        bullish_score: float,
        bearish_score: float,
    ):

        bullish_score = float(
            bullish_score
        )

        bearish_score = float(
            bearish_score
        )

        highest = max(
            bullish_score,
            bearish_score,
        )

        difference = abs(
            bullish_score
            - bearish_score
        )

        # ----------------------------------------------------
        # Not enough evidence
        # ----------------------------------------------------

        if highest < self.min_signal_score:

            return (
                "neutral",
                self._calculate_confidence(
                    highest,
                    difference,
                ),
            )

        # ----------------------------------------------------
        # Conflicting evidence
        # ----------------------------------------------------

        if difference < self.min_direction_edge:

            return (
                "neutral",
                self._calculate_confidence(
                    highest,
                    difference,
                ),
            )

        # ----------------------------------------------------
        # Bullish
        # ----------------------------------------------------

        if bullish_score > bearish_score:

            return (
                "bullish",
                self._calculate_confidence(
                    bullish_score,
                    difference,
                ),
            )

        # ----------------------------------------------------
        # Bearish
        # ----------------------------------------------------

        if bearish_score > bullish_score:

            return (
                "bearish",
                self._calculate_confidence(
                    bearish_score,
                    difference,
                ),
            )

        return (
            "neutral",
            0.0,
        )

    @staticmethod
    def _calculate_confidence(
        score: float,
        difference: float,
    ) -> float:

        # This is a normalized decision confidence,
        # NOT probability of trade success.

        score_component = min(
            1.0,
            max(
                0.0,
                score / 100.0,
            ),
        )

        edge_component = min(
            1.0,
            max(
                0.0,
                difference / 50.0,
            ),
        )

        confidence = (
            score_component * 0.60
            + edge_component * 0.40
        )

        return round(
            confidence * 100.0,
            2,
        )

    # ========================================================
    # EVENT COLLECTION
    # ========================================================

    def _collect_events(
        self,
        current_index: int,
        structure_result: Any,
        liquidity_result: Any,
        fvg_result: Any,
        order_block_result: Any,
        zone_result: Any,
    ) -> List[SMCEvent]:

        events: List[SMCEvent] = []

        # ----------------------------------------------------
        # Structure events
        # ----------------------------------------------------

        structure_events = _get(
            structure_result,
            "events",
            [],
        )

        for event in structure_events:

            if not self._is_known_at(
                event,
                current_index,
            ):
                continue

            confirmation_index = _safe_int(
                _get(
                    event,
                    "confirmation_index",
                )
            )

            event_index = _safe_int(
                _get(
                    event,
                    "index",
                    confirmation_index,
                )
            )

            events.append(
                SMCEvent(
                    event_type=_event_type(
                        event
                    ),
                    direction=_normalize_direction(
                        _get(
                            event,
                            "direction",
                        )
                    ),
                    index=event_index,
                    confirmation_index=confirmation_index,
                    timestamp=_timestamp(
                        _get(
                            event,
                            "confirmation_timestamp",
                            _get(
                                event,
                                "timestamp",
                            ),
                        )
                    ),
                    source="structure",
                    source_id=(
                        str(
                            _get(
                                event,
                                "id",
                            )
                        )
                        if _get(
                            event,
                            "id",
                        ) is not None
                        else None
                    ),
                    price=_safe_float(
                        _get(
                            event,
                            "price",
                            0.0,
                        )
                    ),
                    score=_safe_float(
                        _get(
                            event,
                            "score",
                            0.0,
                        )
                    ),
                )
            )

        # ----------------------------------------------------
        # Liquidity
        # ----------------------------------------------------

        sweeps = _get(
            liquidity_result,
            "sweeps",
            [],
        )

        for sweep in sweeps:

            if not self._is_known_at(
                sweep,
                current_index,
            ):
                continue

            confirmation_index = _safe_int(
                _get(
                    sweep,
                    "confirmation_index",
                )
            )

            sweep_index = _safe_int(
                _get(
                    sweep,
                    "sweep_index",
                    confirmation_index,
                )
            )

            reaction = _normalize_direction(
                _get(
                    sweep,
                    "reaction",
                    _get(
                        sweep,
                        "direction",
                    ),
                )
            )

            events.append(
                SMCEvent(
                    event_type="LIQUIDITY_SWEEP",
                    direction=reaction,
                    index=sweep_index,
                    confirmation_index=confirmation_index,
                    timestamp=_timestamp(
                        _get(
                            sweep,
                            "confirmation_timestamp",
                            _get(
                                sweep,
                                "sweep_timestamp",
                            ),
                        )
                    ),
                    source="liquidity",
                    source_id=(
                        str(
                            _get(
                                sweep,
                                "id",
                                _get(
                                    sweep,
                                    "source_level_id",
                                ),
                            )
                        )
                        if _get(
                            sweep,
                            "id",
                            _get(
                                sweep,
                                "source_level_id",
                            ),
                        ) is not None
                        else None
                    ),
                    price=self._liquidity_price(
                        sweep
                    ),
                    score=_safe_float(
                        _get(
                            sweep,
                            "strength",
                            0.0,
                        )
                    ),
                )
            )

        # ----------------------------------------------------
        # FVG
        # ----------------------------------------------------

        fvgs = _get(
            fvg_result,
            "all_fvgs",
            [],
        )

        for fvg in fvgs:

            if not self._is_known_at(
                fvg,
                current_index,
            ):
                continue

            confirmation_index = _safe_int(
                _get(
                    fvg,
                    "confirmation_index",
                )
            )

            events.append(
                SMCEvent(
                    event_type="FVG",
                    direction=_normalize_direction(
                        _get(
                            fvg,
                            "direction",
                        )
                    ),
                    index=_safe_int(
                        _get(
                            fvg,
                            "index",
                            confirmation_index,
                        )
                    ),
                    confirmation_index=confirmation_index,
                    timestamp=_timestamp(
                        _get(
                            fvg,
                            "confirmation_timestamp",
                        )
                    ),
                    source="fvg",
                    source_id=(
                        str(
                            _get(
                                fvg,
                                "id",
                            )
                        )
                        if _get(
                            fvg,
                            "id",
                        ) is not None
                        else None
                    ),
                    price=self._midpoint(
                        _get(
                            fvg,
                            "gap_low",
                        ),
                        _get(
                            fvg,
                            "gap_high",
                        ),
                    ),
                    score=_safe_float(
                        _get(
                            fvg,
                            "score",
                            0.0,
                        )
                    ),
                )
            )

        # ----------------------------------------------------
        # Order Blocks
        # ----------------------------------------------------

        obs = _get(
            order_block_result,
            "all_order_blocks",
            _get(
                order_block_result,
                "order_blocks",
                [],
            ),
        )

        for ob in obs:

            if not self._is_known_at(
                ob,
                current_index,
            ):
                continue

            confirmation_index = _safe_int(
                _get(
                    ob,
                    "confirmation_index",
                )
            )

            events.append(
                SMCEvent(
                    event_type="ORDER_BLOCK",
                    direction=_normalize_direction(
                        _get(
                            ob,
                            "direction",
                        )
                    ),
                    index=_safe_int(
                        _get(
                            ob,
                            "index",
                            confirmation_index,
                        )
                    ),
                    confirmation_index=confirmation_index,
                    timestamp=_timestamp(
                        _get(
                            ob,
                            "confirmation_timestamp",
                        )
                    ),
                    source="order_block",
                    source_id=(
                        str(
                            _get(
                                ob,
                                "id",
                            )
                        )
                        if _get(
                            ob,
                            "id",
                        ) is not None
                        else None
                    ),
                    price=self._midpoint(
                        _get(
                            ob,
                            "low",
                        ),
                        _get(
                            ob,
                            "high",
                        ),
                    ),
                    score=_safe_float(
                        _get(
                            ob,
                            "score",
                            0.0,
                        )
                    ),
                )
            )

        # ----------------------------------------------------
        # Zones
        # ----------------------------------------------------

        zones = _get(
            zone_result,
            "all_zones",
            [],
        )

        for zone in zones:

            if not self._is_known_at(
                zone,
                current_index,
            ):
                continue

            confirmation_index = _safe_int(
                _get(
                    zone,
                    "confirmation_index",
                )
            )

            events.append(
                SMCEvent(
                    event_type="ZONE",
                    direction=_normalize_direction(
                        _get(
                            zone,
                            "direction",
                        )
                    ),
                    index=confirmation_index,
                    confirmation_index=confirmation_index,
                    timestamp=_timestamp(
                        _get(
                            zone,
                            "confirmation_timestamp",
                        )
                    ),
                    source="zones",
                    source_id=(
                        str(
                            _get(
                                zone,
                                "zone_id",
                            )
                        )
                        if _get(
                            zone,
                            "zone_id",
                        ) is not None
                        else None
                    ),
                    price=self._midpoint(
                        _get(
                            zone,
                            "outer_low",
                        ),
                        _get(
                            zone,
                            "outer_high",
                        ),
                    ),
                    score=_safe_float(
                        _get(
                            zone,
                            "score",
                            0.0,
                        )
                    ),
                )
            )

        # ----------------------------------------------------
        # Sort chronologically
        # ----------------------------------------------------

        events.sort(
            key=lambda e: (
                e.confirmation_index,
                e.index,
            )
        )

        return events

    # ========================================================
    # CONFIRMATION HELPERS
    # ========================================================

    @staticmethod
    def _is_known_at(
        obj: Any,
        current_index: int,
    ) -> bool:

        if obj is None:
            return False

        confirmation_index = _get(
            obj,
            "confirmation_index",
            None,
        )

        # ----------------------------------------------------
        # STRICT V2.1:
        #
        # No fallback to:
        #   index
        #   source_index
        #   sweep_index
        #
        # because those can represent formation rather
        # than confirmation.
        # ----------------------------------------------------

        if confirmation_index is None:
            return False

        try:

            confirmation_index = int(
                confirmation_index
            )

        except Exception:

            return False

        return (
            confirmation_index
            <= current_index
        )

    def _confirmed_events(
        self,
        result: Any,
        current_index: int,
    ) -> List[Any]:

        if result is None:
            return []

        events = _get(
            result,
            "events",
            [],
        )

        return [
            event
            for event in events
            if self._is_known_at(
                event,
                current_index,
            )
        ]

    @staticmethod
    def _latest_event(
        events: List[Any],
        event_name: str,
    ) -> Any:

        matching = [
            event
            for event in events
            if event_name
            in _event_type(event)
        ]

        if not matching:
            return None

        return max(
            matching,
            key=lambda e: _safe_int(
                _get(
                    e,
                    "confirmation_index",
                )
            ),
        )

    def _latest_confirmed_sweep(
        self,
        result: Any,
        current_index: int,
    ) -> Any:

        if result is None:
            return None

        sweeps = _get(
            result,
            "sweeps",
            [],
        )

        candidates = [
            sweep
            for sweep in sweeps
            if self._is_known_at(
                sweep,
                current_index,
            )
        ]

        if not candidates:
            return None

        return max(
            candidates,
            key=lambda x: _safe_int(
                _get(
                    x,
                    "confirmation_index",
                )
            ),
        )

    def _best_confirmed_zone(
        self,
        result: Any,
        current_index: int,
    ) -> Any:

        if result is None:
            return None

        zones = _get(
            result,
            "active_zones",
            [],
        )

        candidates = [
            zone
            for zone in zones
            if self._is_known_at(
                zone,
                current_index,
            )
            and not bool(
                _get(
                    zone,
                    "invalidated",
                    False,
                )
            )
        ]

        if not candidates:
            return None

        return max(
            candidates,
            key=lambda z: (
                _safe_float(
                    _get(
                        z,
                        "score",
                        0.0,
                    )
                ),
                _safe_int(
                    _get(
                        z,
                        "confirmation_index",
                    )
                ),
            ),
        )

    # ========================================================
    # MISC HELPERS
    # ========================================================

    @staticmethod
    def _midpoint(
        low: Any,
        high: Any,
    ) -> Optional[float]:

        if low is None or high is None:
            return None

        try:

            return (
                float(low)
                + float(high)
            ) / 2.0

        except Exception:

            return None

    @staticmethod
    def _liquidity_price(
        sweep: Any,
    ) -> Optional[float]:

        level = _get(
            sweep,
            "level",
            None,
        )

        if isinstance(
            level,
            dict,
        ):

            level = level.get(
                "price"
            )

        else:

            price = _get(
                level,
                "price",
                None,
            )

            if price is not None:
                level = price

        if level is None:

            level = _get(
                sweep,
                "level_price",
                None,
            )

        try:

            return float(level)

        except Exception:

            return None

    @staticmethod
    def _row_timestamp(
        row: pd.Series,
    ) -> Optional[datetime]:

        for name in (
            "timestamp",
            "time",
            "datetime",
            "date",
        ):

            if name in row.index:

                value = _timestamp(
                    row[name]
                )

                if value is not None:
                    return value

        return None

    # ========================================================
    # DATAFRAME
    # ========================================================

    @staticmethod
    def _validate_dataframe(
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        if not isinstance(
            df,
            pd.DataFrame,
        ):

            raise TypeError(
                "df must be pandas.DataFrame"
            )

        if df.empty:
            return df.copy()

        data = df.copy()

        # ----------------------------------------------------
        # Case-insensitive column mapping
        # ----------------------------------------------------

        columns = {
            str(c).lower().strip(): c
            for c in data.columns
        }

        required = (
            "open",
            "high",
            "low",
            "close",
        )

        for name in required:

            if name not in columns:

                raise ValueError(
                    f"Missing OHLC column: {name}"
                )

            original = columns[name]

            if original != name:
                data[name] = data[original]

        # ----------------------------------------------------
        # Timestamp
        # ----------------------------------------------------

        timestamp_source = None

        for candidate in (
            "timestamp",
            "time",
            "datetime",
            "date",
        ):

            if candidate in columns:

                timestamp_source = columns[
                    candidate
                ]

                break

        if timestamp_source is not None:

            data["timestamp"] = pd.to_datetime(
                data[timestamp_source],
                errors="coerce",
            )

            data = data.sort_values(
                "timestamp"
            )

        # ----------------------------------------------------
        # Numeric OHLC
        # ----------------------------------------------------

        for name in required:

            data[name] = pd.to_numeric(
                data[name],
                errors="coerce",
            )

        data = data.dropna(
            subset=required
        )

        # ----------------------------------------------------
        # Remove impossible candles
        # ----------------------------------------------------

        data = data[
            (data["high"] >= data["low"])
            & (data["high"] >= data["open"])
            & (data["high"] >= data["close"])
            & (data["low"] <= data["open"])
            & (data["low"] <= data["close"])
        ]

        # ----------------------------------------------------
        # Reset positional index.
        #
        # From this point forward current_index is positional
        # in this normalized dataframe.
        # ----------------------------------------------------

        data = data.reset_index(
            drop=True
        )

        return data


# ============================================================
# CONVENIENCE API
# ============================================================

def analyze_smc(
    df: pd.DataFrame,
    current_index: Optional[int] = None,
    **kwargs,
) -> SMCResult:

    engine = SMCEngine(
        **kwargs
    )

    return engine.analyze(
        df,
        current_index=current_index,
    )


def detect_smc(
    df: pd.DataFrame,
    current_index: Optional[int] = None,
    **kwargs,
) -> SMCResult:

    return analyze_smc(
        df,
        current_index=current_index,
        **kwargs,
    )


# Compatibility aliases
SMCEngineV2 = SMCEngine
SMCEngineV21 = SMCEngine


__all__ = [
    "SMCEvent",
    "SMCContext",
    "SMCResult",
    "SMCEngine",
    "SMCEngineV2",
    "SMCEngineV21",
    "analyze_smc",
    "detect_smc",
]