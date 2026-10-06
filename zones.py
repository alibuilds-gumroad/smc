"""
SMC Zones Engine V2.1
=====================

Purpose
-------
Build causal trading zones from already-detected:

    Structure
        +
    Liquidity Sweep
        +
    Fair Value Gap (FVG)
        +
    Order Block (OB)

This module is a CONFLUENCE layer.
It does NOT redefine/re-detect SMC primitives.

Core guarantees
---------------
1. No future-data leakage.
2. A component is usable only after confirmation_index.
3. Zone lifecycle is evaluated only with candles available at current_index.
4. FVG / OB / Structure / Liquidity APIs are kept separate.
5. Suitable for candle-by-candle backtesting.

Pipeline
--------
Structure
    ↓
Liquidity Sweep
    ↓
FVG / OB
    ↓
Confluence Zone
    ↓
Lifecycle
    ↓
Score
    ↓
Best Zone
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Sequence
from datetime import datetime
import math

import pandas as pd
import numpy as np


# ============================================================
# CONSTANTS
# ============================================================

DEFAULT_LOOKBACK = 150
DEFAULT_ATR_PERIOD = 14

DEFAULT_MERGE_DISTANCE_ATR = 0.20
DEFAULT_MAX_ZONE_ATR = 2.50
DEFAULT_INVALIDATION_ATR = 0.10

DEFAULT_FRESHNESS_BARS = 20
DEFAULT_MIN_SCORE = 20.0

DEFAULT_LIQUIDITY_DISTANCE_ATR = 1.00


# ============================================================
# HELPERS
# ============================================================

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        if math.isfinite(result):
            return result
    except Exception:
        pass
    return default


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _timestamp(value: Any) -> Optional[datetime]:
    if value is None:
        return None

    if isinstance(value, datetime):
        return value

    try:
        ts = pd.Timestamp(value)
        if pd.isna(ts):
            return None
        return ts.to_pydatetime()
    except Exception:
        return None


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default

    if isinstance(obj, dict):
        return obj.get(name, default)

    return getattr(obj, name, default)


def _direction(obj: Any) -> Optional[str]:
    value = _get(obj, "direction")

    if value is None:
        return None

    value = str(value).lower().strip()

    if value in ("bullish", "buy", "long", "up"):
        return "bullish"

    if value in ("bearish", "sell", "short", "down"):
        return "bearish"

    return value


# ============================================================
# ATR
# ============================================================

def _atr(df: pd.DataFrame, period: int = DEFAULT_ATR_PERIOD) -> pd.Series:
    """
    Wilder-style ATR approximation.

    Uses only current and previous candles.
    """

    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    previous_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(
        alpha=1.0 / max(1, period),
        adjust=False,
        min_periods=1,
    ).mean()

    return atr


# ============================================================
# ZONE DATACLASS
# ============================================================

@dataclass
class ConfluenceZone:
    """
    A tradable SMC zone.

    confirmation_index is the MOST IMPORTANT field for
    anti-lookahead validation.
    """

    zone_id: str

    direction: str

    # --------------------------------------------------------
    # Geometry
    # --------------------------------------------------------

    outer_low: float
    outer_high: float

    core_low: float
    core_high: float

    outer_size: float
    core_size: float

    zone_atr: float
    zone_atr_ratio: float

    # --------------------------------------------------------
    # Confirmation
    # --------------------------------------------------------

    confirmation_index: int
    confirmation_timestamp: Optional[datetime]

    # --------------------------------------------------------
    # Components
    # --------------------------------------------------------

    fvg_id: Optional[str] = None
    ob_id: Optional[str] = None

    fvg: Any = None
    order_block: Any = None

    has_fvg: bool = False
    has_order_block: bool = False
    is_confluence: bool = False

    # --------------------------------------------------------
    # Liquidity context
    # --------------------------------------------------------

    liquidity_sweep_id: Optional[str] = None
    liquidity_sweep: Any = None

    liquidity_confirmed: bool = False

    # --------------------------------------------------------
    # Structure context
    # --------------------------------------------------------

    structure_event_id: Optional[str] = None
    structure_event: Any = None

    structure_type: Optional[str] = None
    structure_direction: Optional[str] = None

    structure_confirmed: bool = False

    # --------------------------------------------------------
    # Lifecycle
    # --------------------------------------------------------

    status: str = "active"

    touched: bool = False
    mitigated: bool = False
    filled: bool = False
    invalidated: bool = False

    touch_index: Optional[int] = None
    touch_timestamp: Optional[datetime] = None

    mitigation_index: Optional[int] = None
    mitigation_timestamp: Optional[datetime] = None

    fill_index: Optional[int] = None
    fill_timestamp: Optional[datetime] = None

    invalidation_index: Optional[int] = None
    invalidation_timestamp: Optional[datetime] = None

    # --------------------------------------------------------
    # Score
    # --------------------------------------------------------

    fvg_score: float = 0.0
    ob_score: float = 0.0

    liquidity_score: float = 0.0
    structure_score: float = 0.0

    freshness_score: float = 0.0
    geometry_score: float = 0.0

    confluence_bonus: float = 0.0

    score: float = 0.0

    # --------------------------------------------------------
    # Current context
    # --------------------------------------------------------

    current_index: int = -1
    current_timestamp: Optional[datetime] = None

    def confirmed_by(self, candle_index: int) -> bool:
        """
        Strict causal test.
        """

        return self.confirmation_index <= candle_index

    @property
    def is_active(self) -> bool:
        return self.status == "active"

    @property
    def midpoint(self) -> float:
        return (self.outer_low + self.outer_high) / 2.0

    @property
    def entry_low(self) -> float:
        """
        Conservative entry range.

        Core zone when available, otherwise outer zone.
        """

        if self.core_high > self.core_low:
            return self.core_low

        return self.outer_low

    @property
    def entry_high(self) -> float:
        if self.core_high > self.core_low:
            return self.core_high

        return self.outer_high


# ============================================================
# RESULT
# ============================================================

@dataclass
class ZoneResult:

    detected: bool = False

    latest: Optional[ConfluenceZone] = None
    best: Optional[ConfluenceZone] = None

    active_zones: List[ConfluenceZone] = field(default_factory=list)
    touched_zones: List[ConfluenceZone] = field(default_factory=list)
    mitigated_zones: List[ConfluenceZone] = field(default_factory=list)
    filled_zones: List[ConfluenceZone] = field(default_factory=list)
    invalidated_zones: List[ConfluenceZone] = field(default_factory=list)

    all_zones: List[ConfluenceZone] = field(default_factory=list)

    current_index: int = -1
    current_timestamp: Optional[datetime] = None

    error: Optional[str] = None

    def confirmed_zones(
        self,
        candle_index: Optional[int] = None,
    ) -> List[ConfluenceZone]:
        """
        Return only zones that were already known at candle_index.
        """

        if candle_index is None:
            candle_index = self.current_index

        return [
            zone
            for zone in self.all_zones
            if zone.confirmation_index <= candle_index
        ]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# ENGINE
# ============================================================

class ZoneEngine:
    """
    Causal SMC Zone Engine V2.1.
    """

    def __init__(
        self,
        lookback: int = DEFAULT_LOOKBACK,
        atr_period: int = DEFAULT_ATR_PERIOD,
        merge_distance_atr: float = DEFAULT_MERGE_DISTANCE_ATR,
        max_zone_atr: float = DEFAULT_MAX_ZONE_ATR,
        invalidation_atr: float = DEFAULT_INVALIDATION_ATR,
        freshness_bars: int = DEFAULT_FRESHNESS_BARS,
        min_score: float = DEFAULT_MIN_SCORE,
        liquidity_distance_atr: float = DEFAULT_LIQUIDITY_DISTANCE_ATR,
    ):
        self.lookback = max(20, int(lookback))
        self.atr_period = max(2, int(atr_period))

        self.merge_distance_atr = max(
            0.0,
            float(merge_distance_atr),
        )

        self.max_zone_atr = max(
            0.1,
            float(max_zone_atr),
        )

        self.invalidation_atr = max(
            0.0,
            float(invalidation_atr),
        )

        self.freshness_bars = max(
            1,
            int(freshness_bars),
        )

        self.min_score = float(min_score)

        self.liquidity_distance_atr = max(
            0.0,
            float(liquidity_distance_atr),
        )

    # ========================================================
    # PUBLIC API
    # ========================================================

    def analyze(
        self,
        df: pd.DataFrame,
        current_index: Optional[int] = None,
        fvg_result: Any = None,
        order_block_result: Any = None,
        structure_result: Any = None,
        liquidity_result: Any = None,
    ) -> ZoneResult:

        try:

            data = self._validate_dataframe(df)

            if data.empty:
                return ZoneResult(
                    detected=False,
                    error="Empty dataframe",
                )

            if current_index is None:
                current_index = len(data) - 1

            current_index = int(current_index)

            if current_index < 0:
                return ZoneResult(
                    detected=False,
                    error="current_index < 0",
                )

            if current_index >= len(data):
                current_index = len(data) - 1

            # ------------------------------------------------
            # IMPORTANT:
            # Only data up to current candle is allowed.
            # ------------------------------------------------

            work = data.iloc[: current_index + 1].copy()

            if work.empty:
                return ZoneResult(
                    detected=False,
                    error="No candles available",
                )

            atr_series = _atr(
                work,
                period=self.atr_period,
            )

            current_atr = _safe_float(
                atr_series.iloc[-1],
                0.0,
            )

            if current_atr <= 0:
                current_atr = max(
                    _safe_float(
                        work["high"].iloc[-1]
                        - work["low"].iloc[-1],
                    ),
                    1e-9,
                )

            # ------------------------------------------------
            # If upstream results are not supplied, run them.
            # ------------------------------------------------

            if fvg_result is None:
                fvg_result = self._run_fvg(
                    work,
                    current_index=len(work) - 1,
                )

            if order_block_result is None:
                order_block_result = self._run_order_block(
                    work,
                    current_index=len(work) - 1,
                    structure_result=structure_result,
                    liquidity_result=liquidity_result,
                )

            if structure_result is None:
                structure_result = self._run_structure(
                    work,
                    current_index=len(work) - 1,
                )

            if liquidity_result is None:
                liquidity_result = self._run_liquidity(
                    work,
                    current_index=len(work) - 1,
                )

            # ------------------------------------------------
            # Extract confirmed components
            # ------------------------------------------------

            fvgs = self._confirmed_fvgs(
                fvg_result,
                current_index=len(work) - 1,
            )

            obs = self._confirmed_obs(
                order_block_result,
                current_index=len(work) - 1,
            )

            # ------------------------------------------------
            # Build candidate zones
            # ------------------------------------------------

            zones = self._build_zones(
                fvgs=fvgs,
                obs=obs,
                work=work,
                atr=current_atr,
            )

            # ------------------------------------------------
            # Attach context
            # ------------------------------------------------

            for zone in zones:

                self._attach_structure(
                    zone,
                    structure_result,
                    current_index=len(work) - 1,
                )

                self._attach_liquidity(
                    zone,
                    liquidity_result,
                    current_index=len(work) - 1,
                    atr=current_atr,
                )

                self._calculate_score(
                    zone,
                    current_index=len(work) - 1,
                )

            # ------------------------------------------------
            # Filter oversized / low-quality zones
            # ------------------------------------------------

            zones = [
                zone
                for zone in zones
                if zone.score >= self.min_score
                and zone.zone_atr_ratio <= self.max_zone_atr
            ]

            # ------------------------------------------------
            # Lifecycle
            # ------------------------------------------------

            for zone in zones:

                self._update_lifecycle(
                    zone,
                    work,
                )

                zone.current_index = current_index
                zone.current_timestamp = self._row_timestamp(
                    work.iloc[-1]
                )

            # ------------------------------------------------
            # Deduplicate
            # ------------------------------------------------

            zones = self._deduplicate_zones(zones)

            # ------------------------------------------------
            # Sort
            # ------------------------------------------------

            zones.sort(
                key=lambda z: (
                    z.confirmation_index,
                    z.score,
                ),
                reverse=True,
            )

            active = [
                z for z in zones
                if z.status == "active"
            ]

            touched = [
                z for z in zones
                if z.touched
                and not z.invalidated
            ]

            mitigated = [
                z for z in zones
                if z.mitigated
                and not z.invalidated
            ]

            filled = [
                z for z in zones
                if z.filled
                and not z.invalidated
            ]

            invalidated = [
                z for z in zones
                if z.invalidated
            ]

            best = self._select_best(active)

            latest = (
                max(
                    zones,
                    key=lambda z: z.confirmation_index,
                )
                if zones
                else None
            )

            return ZoneResult(
                detected=bool(zones),
                latest=latest,
                best=best,
                active_zones=active,
                touched_zones=touched,
                mitigated_zones=mitigated,
                filled_zones=filled,
                invalidated_zones=invalidated,
                all_zones=zones,
                current_index=current_index,
                current_timestamp=self._row_timestamp(
                    work.iloc[-1]
                ),
            )

        except Exception as exc:

            return ZoneResult(
                detected=False,
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
    # UPSTREAM MODULES
    # ========================================================

    def _run_fvg(
        self,
        df: pd.DataFrame,
        current_index: int,
    ):

        from .fvg import FVGEngine

        engine = FVGEngine(
            lookback=self.lookback,
            atr_period=self.atr_period,
        )

        return engine.analyze(
            df,
            current_index=current_index,
        )

    def _run_order_block(
        self,
        df: pd.DataFrame,
        current_index: int,
        structure_result: Any = None,
        liquidity_result: Any = None,
    ):

        from .order_block import OrderBlockEngine

        engine = OrderBlockEngine(
            lookback=self.lookback,
            atr_period=self.atr_period,
        )

        return engine.analyze(
            df,
            current_index=current_index,
            structure_result=structure_result,
            liquidity_result=liquidity_result,
        )

    def _run_structure(
        self,
        df: pd.DataFrame,
        current_index: int,
    ):

        from .structure import MarketStructureEngine

        engine = MarketStructureEngine()

        return engine.analyze(
            df,
            current_index=current_index,
        )

    def _run_liquidity(
        self,
        df: pd.DataFrame,
        current_index: int,
    ):

        from .liquidity import LiquidityEngine

        engine = LiquidityEngine()

        return engine.analyze(
            df,
            current_index=current_index,
        )

    # ========================================================
    # COMPONENT EXTRACTION
    # ========================================================

    def _confirmed_fvgs(
        self,
        result: Any,
        current_index: int,
    ) -> List[Any]:

        if result is None:
            return []

        candidates = _get(
            result,
            "all_fvgs",
            None,
        )

        if candidates is None:
            candidates = []

            for name in (
                "bullish_fvgs",
                "bearish_fvgs",
            ):
                values = _get(
                    result,
                    name,
                    [],
                )

                if values:
                    candidates.extend(values)

        output = []

        for fvg in candidates:

            confirmation_index = _get(
                fvg,
                "confirmation_index",
                None,
            )

            # Strict V2.1:
            # NO fallback to formation/index.
            if confirmation_index is None:
                continue

            if int(confirmation_index) <= current_index:
                output.append(fvg)

        return self._deduplicate_objects(
            output
        )

    def _confirmed_obs(
        self,
        result: Any,
        current_index: int,
    ) -> List[Any]:

        if result is None:
            return []

        candidates = _get(
            result,
            "all_order_blocks",
            None,
        )

        if candidates is None:
            candidates = _get(
                result,
                "order_blocks",
                [],
            )

        output = []

        for ob in candidates:

            confirmation_index = _get(
                ob,
                "confirmation_index",
                None,
            )

            # Strict V2.1:
            # NO fallback to source index.
            if confirmation_index is None:
                continue

            if int(confirmation_index) <= current_index:
                output.append(ob)

        return self._deduplicate_objects(
            output
        )

    # ========================================================
    # BUILD ZONES
    # ========================================================

    def _build_zones(
        self,
        fvgs: Sequence[Any],
        obs: Sequence[Any],
        work: pd.DataFrame,
        atr: float,
    ) -> List[ConfluenceZone]:

        zones: List[ConfluenceZone] = []

        # ----------------------------------------------------
        # FVG-only zones
        # ----------------------------------------------------

        for fvg in fvgs:

            direction = _direction(fvg)

            if direction not in (
                "bullish",
                "bearish",
            ):
                continue

            low, high = self._component_range(
                fvg,
                "fvg",
            )

            if high <= low:
                continue

            confirmation_index = _safe_int(
                _get(
                    fvg,
                    "confirmation_index",
                )
            )

            if confirmation_index < 0:
                continue

            zone = self._create_zone(
                direction=direction,
                outer_low=low,
                outer_high=high,
                core_low=low,
                core_high=high,
                confirmation_index=confirmation_index,
                confirmation_timestamp=_get(
                    fvg,
                    "confirmation_timestamp",
                ),
                fvg=fvg,
                ob=None,
                atr=atr,
            )

            zones.append(zone)

        # ----------------------------------------------------
        # OB-only zones
        # ----------------------------------------------------

        for ob in obs:

            direction = _direction(ob)

            if direction not in (
                "bullish",
                "bearish",
            ):
                continue

            low, high = self._component_range(
                ob,
                "ob",
            )

            if high <= low:
                continue

            confirmation_index = _safe_int(
                _get(
                    ob,
                    "confirmation_index",
                )
            )

            if confirmation_index < 0:
                continue

            zone = self._create_zone(
                direction=direction,
                outer_low=low,
                outer_high=high,
                core_low=low,
                core_high=high,
                confirmation_index=confirmation_index,
                confirmation_timestamp=_get(
                    ob,
                    "confirmation_timestamp",
                ),
                fvg=None,
                ob=ob,
                atr=atr,
            )

            zones.append(zone)

        # ----------------------------------------------------
        # FVG + OB confluence
        # ----------------------------------------------------

        for fvg in fvgs:

            fvg_direction = _direction(fvg)

            if fvg_direction not in (
                "bullish",
                "bearish",
            ):
                continue

            fvg_low, fvg_high = self._component_range(
                fvg,
                "fvg",
            )

            fvg_confirmation = _safe_int(
                _get(
                    fvg,
                    "confirmation_index",
                )
            )

            for ob in obs:

                if _direction(ob) != fvg_direction:
                    continue

                ob_confirmation = _safe_int(
                    _get(
                        ob,
                        "confirmation_index",
                    )
                )

                if fvg_confirmation < 0:
                    continue

                if ob_confirmation < 0:
                    continue

                ob_low, ob_high = self._component_range(
                    ob,
                    "ob",
                )

                if ob_high <= ob_low:
                    continue

                # ------------------------------------------------
                # Components must already be known when zone
                # becomes confirmed.
                # ------------------------------------------------

                confirmation_index = max(
                    fvg_confirmation,
                    ob_confirmation,
                )

                if confirmation_index >= len(work):
                    continue

                # ------------------------------------------------
                # Spatial relation
                # ------------------------------------------------

                distance = self._range_distance(
                    fvg_low,
                    fvg_high,
                    ob_low,
                    ob_high,
                )

                if distance > (
                    self.merge_distance_atr * atr
                ):
                    continue

                outer_low = min(
                    fvg_low,
                    ob_low,
                )

                outer_high = max(
                    fvg_high,
                    ob_high,
                )

                core_low = max(
                    fvg_low,
                    ob_low,
                )

                core_high = min(
                    fvg_high,
                    ob_high,
                )

                # ------------------------------------------------
                # If ranges do not overlap but are close,
                # create a narrow entry core around midpoint
                # rather than pretending there is a true overlap.
                # ------------------------------------------------

                if core_high <= core_low:

                    nearest = self._nearest_points(
                        fvg_low,
                        fvg_high,
                        ob_low,
                        ob_high,
                    )

                    core_low = nearest
                    core_high = nearest

                zone = self._create_zone(
                    direction=fvg_direction,
                    outer_low=outer_low,
                    outer_high=outer_high,
                    core_low=core_low,
                    core_high=core_high,
                    confirmation_index=confirmation_index,
                    confirmation_timestamp=self._later_timestamp(
                        _get(
                            fvg,
                            "confirmation_timestamp",
                        ),
                        _get(
                            ob,
                            "confirmation_timestamp",
                        ),
                    ),
                    fvg=fvg,
                    ob=ob,
                    atr=atr,
                )

                zones.append(zone)

        return zones

    # ========================================================
    # ZONE CREATION
    # ========================================================

    def _create_zone(
        self,
        direction: str,
        outer_low: float,
        outer_high: float,
        core_low: float,
        core_high: float,
        confirmation_index: int,
        confirmation_timestamp: Any,
        fvg: Any,
        ob: Any,
        atr: float,
    ) -> ConfluenceZone:

        outer_low = float(outer_low)
        outer_high = float(outer_high)

        if outer_low > outer_high:
            outer_low, outer_high = (
                outer_high,
                outer_low,
            )

        core_low = float(core_low)
        core_high = float(core_high)

        if core_low > core_high:
            core_low, core_high = (
                core_high,
                core_low,
            )

        outer_size = max(
            0.0,
            outer_high - outer_low,
        )

        core_size = max(
            0.0,
            core_high - core_low,
        )

        zone_atr = max(
            float(atr),
            1e-9,
        )

        zone_atr_ratio = (
            outer_size / zone_atr
        )

        fvg_id = (
            str(_get(fvg, "id"))
            if fvg is not None
            and _get(fvg, "id") is not None
            else None
        )

        ob_id = (
            str(_get(ob, "id"))
            if ob is not None
            and _get(ob, "id") is not None
            else None
        )

        zone_id = self._make_zone_id(
            direction,
            fvg_id,
            ob_id,
            confirmation_index,
            outer_low,
            outer_high,
        )

        return ConfluenceZone(
            zone_id=zone_id,
            direction=direction,

            outer_low=outer_low,
            outer_high=outer_high,

            core_low=core_low,
            core_high=core_high,

            outer_size=outer_size,
            core_size=core_size,

            zone_atr=zone_atr,
            zone_atr_ratio=zone_atr_ratio,

            confirmation_index=confirmation_index,
            confirmation_timestamp=_timestamp(
                confirmation_timestamp
            ),

            fvg_id=fvg_id,
            ob_id=ob_id,

            fvg=fvg,
            order_block=ob,

            has_fvg=fvg is not None,
            has_order_block=ob is not None,
            is_confluence=(
                fvg is not None
                and ob is not None
            ),
        )

    # ========================================================
    # STRUCTURE CONTEXT
    # ========================================================

    def _attach_structure(
        self,
        zone: ConfluenceZone,
        structure_result: Any,
        current_index: int,
    ):

        if structure_result is None:
            return

        events = _get(
            structure_result,
            "events",
            [],
        )

        if not events:
            return

        candidates = []

        for event in events:

            event_confirmation = _get(
                event,
                "confirmation_index",
                None,
            )

            # Strict:
            # structure event without confirmation is ignored.
            if event_confirmation is None:
                continue

            event_confirmation = int(
                event_confirmation
            )

            if event_confirmation > zone.confirmation_index:
                continue

            if event_confirmation > current_index:
                continue

            direction = _direction(event)

            if direction != zone.direction:
                continue

            candidates.append(event)

        if not candidates:
            return

        event = max(
            candidates,
            key=lambda x: _safe_int(
                _get(
                    x,
                    "confirmation_index",
                )
            ),
        )

        zone.structure_event = event

        zone.structure_event_id = (
            str(_get(event, "id"))
            if _get(event, "id") is not None
            else None
        )

        zone.structure_type = (
            str(
                _get(
                    event,
                    "event_type",
                    _get(
                        event,
                        "type",
                        "",
                    ),
                )
            ).upper()
            or None
        )

        zone.structure_direction = _direction(
            event
        )

        zone.structure_confirmed = True

    # ========================================================
    # LIQUIDITY CONTEXT
    # ========================================================

    def _attach_liquidity(
        self,
        zone: ConfluenceZone,
        liquidity_result: Any,
        current_index: int,
        atr: float,
    ):

        if liquidity_result is None:
            return

        sweeps = _get(
            liquidity_result,
            "sweeps",
            [],
        )

        if not sweeps:
            latest = _get(
                liquidity_result,
                "latest",
            )

            if latest is not None:
                sweeps = [latest]

        candidates = []

        for sweep in sweeps:

            confirmation_index = _get(
                sweep,
                "confirmation_index",
                None,
            )

            # Strict V2.1:
            # sweep MUST explicitly expose confirmation_index.
            if confirmation_index is None:
                continue

            confirmation_index = int(
                confirmation_index
            )

            if confirmation_index > zone.confirmation_index:
                continue

            if confirmation_index > current_index:
                continue

            reaction = _direction(
                sweep
            )

            if reaction != zone.direction:
                continue

            level = self._liquidity_price(
                sweep
            )

            if level is None:
                continue

            distance = self._distance_to_range(
                level,
                zone.outer_low,
                zone.outer_high,
            )

            if distance > (
                self.liquidity_distance_atr
                * atr
            ):
                continue

            candidates.append(
                (
                    confirmation_index,
                    distance,
                    sweep,
                )
            )

        if not candidates:
            return

        candidates.sort(
            key=lambda x: (
                x[0],
                -x[1],
            ),
            reverse=True,
        )

        _, distance, sweep = candidates[0]

        zone.liquidity_sweep = sweep

        zone.liquidity_sweep_id = (
            str(_get(sweep, "id"))
            if _get(sweep, "id") is not None
            else str(
                _get(
                    sweep,
                    "source_level_id",
                    "",
                )
            )
            or None
        )

        zone.liquidity_confirmed = True

        zone.liquidity_score = 10.0

        sweep_strength = _safe_float(
            _get(
                sweep,
                "strength",
                0.0,
            )
        )

        if sweep_strength >= 4.0:
            zone.liquidity_score += 5.0

        zone.liquidity_score = min(
            zone.liquidity_score,
            15.0,
        )

    # ========================================================
    # SCORE
    # ========================================================

    def _calculate_score(
        self,
        zone: ConfluenceZone,
        current_index: int,
    ):

        # ----------------------------------------------------
        # FVG
        # ----------------------------------------------------

        if zone.fvg is not None:

            zone.fvg_score = min(
                30.0,
                max(
                    0.0,
                    _safe_float(
                        _get(
                            zone.fvg,
                            "score",
                            0.0,
                        )
                    )
                    * 0.30,
                ),
            )

        # ----------------------------------------------------
        # OB
        # ----------------------------------------------------

        if zone.order_block is not None:

            zone.ob_score = min(
                30.0,
                max(
                    0.0,
                    _safe_float(
                        _get(
                            zone.order_block,
                            "score",
                            0.0,
                        )
                    )
                    * 0.30,
                ),
            )

        # ----------------------------------------------------
        # Confluence
        # ----------------------------------------------------

        if zone.is_confluence:
            zone.confluence_bonus = 15.0

        # ----------------------------------------------------
        # Structure
        # ----------------------------------------------------

        if zone.structure_confirmed:

            structure_type = (
                str(
                    zone.structure_type
                    or ""
                )
                .upper()
            )

            if "CHOCH" in structure_type:
                zone.structure_score = 10.0

            elif "MSS" in structure_type:
                zone.structure_score = 8.0

            elif "BOS" in structure_type:
                zone.structure_score = 6.0

            else:
                zone.structure_score = 3.0

        # ----------------------------------------------------
        # Freshness
        # ----------------------------------------------------

        age = max(
            0,
            current_index
            - zone.confirmation_index,
        )

        if age <= self.freshness_bars:

            zone.freshness_score = (
                10.0
                * (
                    1.0
                    - (
                        age
                        / max(
                            1,
                            self.freshness_bars,
                        )
                    )
                )
            )

        else:
            zone.freshness_score = 0.0

        # ----------------------------------------------------
        # Geometry
        # ----------------------------------------------------

        ratio = zone.zone_atr_ratio

        if ratio <= 0.25:
            zone.geometry_score = 10.0

        elif ratio <= 0.50:
            zone.geometry_score = 8.0

        elif ratio <= 1.00:
            zone.geometry_score = 6.0

        elif ratio <= 1.50:
            zone.geometry_score = 3.0

        else:
            zone.geometry_score = 0.0

        # ----------------------------------------------------
        # Final score
        # ----------------------------------------------------

        total = (
            zone.fvg_score
            + zone.ob_score
            + zone.confluence_bonus
            + zone.liquidity_score
            + zone.structure_score
            + zone.freshness_score
            + zone.geometry_score
        )

        zone.score = round(
            min(
                100.0,
                max(
                    0.0,
                    total,
                ),
            ),
            2,
        )

    # ========================================================
    # LIFECYCLE
    # ========================================================

    def _update_lifecycle(
        self,
        zone: ConfluenceZone,
        df: pd.DataFrame,
    ):

        start = (
            zone.confirmation_index + 1
        )

        if start >= len(df):
            zone.status = "active"
            return

        for i in range(
            start,
            len(df),
        ):

            row = df.iloc[i]

            high = _safe_float(
                row["high"]
            )

            low = _safe_float(
                row["low"]
            )

            close = _safe_float(
                row["close"]
            )

            timestamp = self._row_timestamp(
                row
            )

            buffer = (
                zone.zone_atr
                * self.invalidation_atr
            )

            # ------------------------------------------------
            # Invalidation
            # ------------------------------------------------

            if zone.direction == "bullish":

                if close < (
                    zone.outer_low
                    - buffer
                ):

                    zone.invalidated = True
                    zone.status = "invalidated"

                    zone.invalidation_index = i
                    zone.invalidation_timestamp = timestamp

                    break

            else:

                if close > (
                    zone.outer_high
                    + buffer
                ):

                    zone.invalidated = True
                    zone.status = "invalidated"

                    zone.invalidation_index = i
                    zone.invalidation_timestamp = timestamp

                    break

            # ------------------------------------------------
            # Touch
            # ------------------------------------------------

            overlaps = (
                high >= zone.outer_low
                and low <= zone.outer_high
            )

            if overlaps:

                if not zone.touched:

                    zone.touched = True
                    zone.status = "touched"

                    zone.touch_index = i
                    zone.touch_timestamp = timestamp

            # ------------------------------------------------
            # Mitigation
            # ------------------------------------------------

            if zone.direction == "bullish":

                penetration = (
                    zone.outer_high
                    - max(
                        low,
                        zone.outer_low,
                    )
                )

            else:

                penetration = (
                    min(
                        high,
                        zone.outer_high,
                    )
                    - zone.outer_low
                )

            penetration = max(
                0.0,
                penetration,
            )

            if (
                zone.outer_size > 0
                and penetration
                >= zone.outer_size * 0.50
            ):

                if not zone.mitigated:

                    zone.mitigated = True
                    zone.status = "mitigated"

                    zone.mitigation_index = i
                    zone.mitigation_timestamp = timestamp

            # ------------------------------------------------
            # Full fill
            # ------------------------------------------------

            if zone.direction == "bullish":

                if low <= zone.outer_low:

                    zone.filled = True
                    zone.status = "filled"

                    zone.fill_index = i
                    zone.fill_timestamp = timestamp

                    break

            else:

                if high >= zone.outer_high:

                    zone.filled = True
                    zone.status = "filled"

                    zone.fill_index = i
                    zone.fill_timestamp = timestamp

                    break

    # ========================================================
    # BEST ZONE
    # ========================================================

    def _select_best(
        self,
        zones: Sequence[ConfluenceZone],
    ) -> Optional[ConfluenceZone]:

        if not zones:
            return None

        return max(
            zones,
            key=lambda z: (
                z.score,
                int(z.is_confluence),
                int(z.liquidity_confirmed),
                int(z.structure_confirmed),
                z.confirmation_index,
            ),
        )

    # ========================================================
    # RANGE HELPERS
    # ========================================================

    def _component_range(
        self,
        obj: Any,
        component_type: str,
    ):

        if component_type == "fvg":

            low = _get(
                obj,
                "gap_low",
            )

            high = _get(
                obj,
                "gap_high",
            )

        else:

            low = _get(
                obj,
                "low",
            )

            high = _get(
                obj,
                "high",
            )

        if low is None or high is None:
            return 0.0, 0.0

        low = float(low)
        high = float(high)

        return (
            min(low, high),
            max(low, high),
        )

    @staticmethod
    def _range_distance(
        low1: float,
        high1: float,
        low2: float,
        high2: float,
    ) -> float:

        if high1 >= low2 and high2 >= low1:
            return 0.0

        if high1 < low2:
            return low2 - high1

        return low1 - high2

    @staticmethod
    def _distance_to_range(
        price: float,
        low: float,
        high: float,
    ) -> float:

        if low <= price <= high:
            return 0.0

        if price < low:
            return low - price

        return price - high

    @staticmethod
    def _nearest_points(
        low1: float,
        high1: float,
        low2: float,
        high2: float,
    ) -> float:

        candidates = [
            abs(low1 - high2),
            abs(high1 - low2),
        ]

        if abs(low1 - high2) <= abs(
            high1 - low2
        ):
            return (
                low1 + high2
            ) / 2.0

        return (
            high1 + low2
        ) / 2.0

    @staticmethod
    def _liquidity_price(
        sweep: Any,
    ) -> Optional[float]:

        level = _get(
            sweep,
            "level",
        )

        if isinstance(level, dict):
            level = level.get(
                "price"
            )

        else:
            level_price = _get(
                level,
                "price",
                None,
            )

            if level_price is not None:
                level = level_price

        if level is None:
            level = _get(
                sweep,
                "level_price",
                None,
            )

        if level is None:
            return None

        try:
            return float(level)
        except Exception:
            return None

    # ========================================================
    # DEDUPLICATION
    # ========================================================

    @staticmethod
    def _deduplicate_objects(
        objects: Sequence[Any],
    ) -> List[Any]:

        result = []
        seen = set()

        for obj in objects:

            object_id = _get(
                obj,
                "id",
                None,
            )

            if object_id is None:
                object_id = id(obj)

            object_id = str(
                object_id
            )

            if object_id in seen:
                continue

            seen.add(object_id)
            result.append(obj)

        return result

    @staticmethod
    def _deduplicate_zones(
        zones: Sequence[ConfluenceZone],
    ) -> List[ConfluenceZone]:

        result = []
        seen = set()

        for zone in zones:

            key = (
                zone.direction,
                zone.fvg_id,
                zone.ob_id,
            )

            # Fallback for component-less duplicates.
            if (
                zone.fvg_id is None
                and zone.ob_id is None
            ):
                key = (
                    zone.direction,
                    zone.confirmation_index,
                    round(
                        zone.outer_low,
                        6,
                    ),
                    round(
                        zone.outer_high,
                        6,
                    ),
                )

            if key in seen:
                continue

            seen.add(key)
            result.append(zone)

        return result

    # ========================================================
    # ID
    # ========================================================

    @staticmethod
    def _make_zone_id(
        direction: str,
        fvg_id: Optional[str],
        ob_id: Optional[str],
        confirmation_index: int,
        low: float,
        high: float,
    ) -> str:

        return (
            f"ZONE_"
            f"{direction.upper()}_"
            f"{fvg_id or 'NOFVG'}_"
            f"{ob_id or 'NOOB'}_"
            f"{confirmation_index}_"
            f"{low:.5f}_"
            f"{high:.5f}"
        )

    # ========================================================
    # DATAFRAME VALIDATION
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
        # Normalize columns
        # ----------------------------------------------------

        lower_map = {
            str(c).lower().strip(): c
            for c in data.columns
        }

        required = [
            "open",
            "high",
            "low",
            "close",
        ]

        for col in required:

            if col not in lower_map:
                raise ValueError(
                    f"Missing OHLC column: {col}"
                )

            original = lower_map[col]

            if original != col:
                data[col] = data[original]

        # ----------------------------------------------------
        # Timestamp
        # ----------------------------------------------------

        timestamp_col = None

        for candidate in (
            "timestamp",
            "time",
            "datetime",
            "date",
        ):

            if candidate in lower_map:
                timestamp_col = lower_map[
                    candidate
                ]
                break

        if timestamp_col is not None:

            data["timestamp"] = pd.to_datetime(
                data[timestamp_col],
                errors="coerce",
            )

            data = data.sort_values(
                "timestamp"
            )

        # ----------------------------------------------------
        # Numeric OHLC
        # ----------------------------------------------------

        for col in required:
            data[col] = pd.to_numeric(
                data[col],
                errors="coerce",
            )

        data = data.dropna(
            subset=required
        )

        # ----------------------------------------------------
        # Reset positional index.
        #
        # IMPORTANT:
        # All confirmation_index values in this module
        # are positional indices of this normalized dataframe.
        # ----------------------------------------------------

        data = data.reset_index(
            drop=True
        )

        return data

    # ========================================================
    # TIMESTAMP
    # ========================================================

    @staticmethod
    def _row_timestamp(
        row: pd.Series,
    ) -> Optional[datetime]:

        for column in (
            "timestamp",
            "time",
            "datetime",
            "date",
        ):

            if column in row.index:

                value = _timestamp(
                    row[column]
                )

                if value is not None:
                    return value

        return None

    @staticmethod
    def _later_timestamp(
        first: Any,
        second: Any,
    ) -> Optional[datetime]:

        first = _timestamp(first)
        second = _timestamp(second)

        if first is None:
            return second

        if second is None:
            return first

        return max(
            first,
            second,
        )


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def detect_zones(
    df: pd.DataFrame,
    current_index: Optional[int] = None,
    **kwargs,
) -> ZoneResult:

    engine = ZoneEngine(
        **kwargs
    )

    return engine.analyze(
        df,
        current_index=current_index,
    )


def analyze_zones(
    df: pd.DataFrame,
    current_index: Optional[int] = None,
    **kwargs,
) -> ZoneResult:

    return detect_zones(
        df,
        current_index=current_index,
        **kwargs,
    )


# Compatibility alias
ZoneEngineV2 = ZoneEngine
ZoneEngineV21 = ZoneEngine


__all__ = [
    "ConfluenceZone",
    "ZoneResult",
    "ZoneEngine",
    "ZoneEngineV2",
    "ZoneEngineV21",
    "detect_zones",
    "analyze_zones",
]