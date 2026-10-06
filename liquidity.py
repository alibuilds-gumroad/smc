"""
smc/liquidity.py
===========================================================
Liquidity Engine V2.1
===========================================================

Purpose
-------
Detect liquidity pools and liquidity sweeps causally.

Design principles
-----------------
1. NO LOOK-AHEAD
   A swing is usable only after its confirmation candle.

2. Correct liquidity terminology
   - Above swing high  -> BUY-SIDE liquidity
   - Below swing low   -> SELL-SIDE liquidity

3. Separate:
   - liquidity_side
   - reaction direction

4. Equal High / Equal Low detection
   ATR-normalized tolerance.

5. Sweep confirmation
   Default behavior:
   price must trade beyond the liquidity level and close back
   inside the level.

6. Displacement is CONFIRMATION, not a requirement.
   A sweep can exist without displacement.

7. ATR-normalized wick / volume / displacement metrics.

8. Every detected object carries:
   - source_index
   - confirmation_index
   - sweep_index

9. Compatible with:
   - structure.py V2.1
   - zones.py V2.1
   - engine.py V2.1
===========================================================
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple
import math

import numpy as np
import pandas as pd


# =========================================================
# DEFAULTS
# =========================================================

DEFAULT_SWING_STRENGTH = 5
DEFAULT_LOOKBACK = 200
DEFAULT_SCAN_RECENT = 30

DEFAULT_ATR_PERIOD = 14

DEFAULT_EQUAL_TOLERANCE_ATR = 0.05

DEFAULT_MIN_WICK_ATR = 0.05
DEFAULT_STRONG_WICK_ATR = 0.50

DEFAULT_VOLUME_LOOKBACK = 20
DEFAULT_VOLUME_RATIO = 1.50

DEFAULT_MIN_STRENGTH = 1

DEFAULT_REQUIRE_CLOSE_BACK = True
DEFAULT_REQUIRE_DISPLACEMENT = False

DEFAULT_DISPLACEMENT_ATR = 0.50


# =========================================================
# HELPERS
# =========================================================

def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Safely read an attribute or dict field."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default

        result = float(value)

        if not math.isfinite(result):
            return default

        return result

    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _timestamp(value: Any) -> Any:
    """Return timestamp in a safe serializable form."""
    if value is None:
        return None

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    try:
        return pd.Timestamp(value).isoformat()
    except Exception:
        return str(value)


def _normalize_direction(value: Any) -> Optional[str]:
    if value is None:
        return None

    value = str(value).strip().lower()

    mapping = {
        "bullish": "bullish",
        "bull": "bullish",
        "buy": "bullish",
        "long": "bullish",

        "bearish": "bearish",
        "bear": "bearish",
        "sell": "bearish",
        "short": "bearish",
    }

    return mapping.get(value)


def _normalize_liquidity_side(value: Any) -> Optional[str]:
    if value is None:
        return None

    value = str(value).strip().lower()

    if value in {
        "buy_side",
        "buy-side",
        "buyside",
        "buy side",
    }:
        return "buy_side"

    if value in {
        "sell_side",
        "sell-side",
        "sellside",
        "sell side",
    }:
        return "sell_side"

    return None


# =========================================================
# ATR
# =========================================================

def _calculate_atr(
    df: pd.DataFrame,
    period: int = DEFAULT_ATR_PERIOD,
) -> pd.Series:
    """
    Wilder-style ATR approximation.

    Uses only current and historical candles.
    """

    high = df["high"]
    low = df["low"]
    close = df["close"]

    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(
        alpha=1.0 / max(int(period), 1),
        adjust=False,
        min_periods=1,
    ).mean()

    return atr


# =========================================================
# DATA CLASSES
# =========================================================

@dataclass
class LiquidityLevel:
    """
    A liquidity pool.

    Important:
    confirmation_index tells us when this level became
    legitimately usable by the strategy.
    """

    id: str

    liquidity_side: str
    liquidity_type: str

    price: float

    source_index: int
    source_timestamp: Any

    confirmation_index: int
    confirmation_timestamp: Any

    source_index_2: Optional[int] = None
    source_timestamp_2: Any = None

    strength: float = 1.0

    atr: float = 0.0

    equal_cluster_size: int = 1

    swept: bool = False
    swept_index: Optional[int] = None
    swept_timestamp: Any = None

    @property
    def direction(self) -> str:
        """
        Expected reaction after sweep.

        Buy-side liquidity above highs:
            bearish reaction

        Sell-side liquidity below lows:
            bullish reaction
        """

        if self.liquidity_side == "buy_side":
            return "bearish"

        return "bullish"

    @property
    def index(self) -> int:
        """Backward-compatible alias."""
        return self.source_index

    @property
    def timestamp(self) -> Any:
        """Backward-compatible alias."""
        return self.source_timestamp


@dataclass
class LiquiditySweep:
    """
    A liquidity sweep event.
    """

    id: str

    detected: bool

    status: str

    liquidity_side: str

    reaction: str

    liquidity_type: str

    level: float

    extreme: float
    close: float

    wick_extension: float
    wick_atr_ratio: float

    volume_ratio: float

    strength: float

    sweep_index: int
    sweep_timestamp: Any

    source_index: int
    source_timestamp: Any

    confirmation_index: int
    confirmation_timestamp: Any

    source_level_id: str

    close_back_confirmed: bool

    displacement_confirmed: bool

    displacement_index: Optional[int] = None
    displacement_timestamp: Any = None

    displacement_atr_ratio: float = 0.0

    atr: float = 0.0

    @property
    def direction(self) -> str:
        """Reaction direction."""
        return self.reaction

    @property
    def index(self) -> int:
        """Backward-compatible alias."""
        return self.sweep_index

    @property
    def timestamp(self) -> Any:
        """Backward-compatible alias."""
        return self.sweep_timestamp


@dataclass
class LiquidityResult:
    detected: bool

    latest: Optional[LiquiditySweep]

    sweeps: List[LiquiditySweep]

    levels: List[LiquidityLevel]

    buy_side_levels: List[LiquidityLevel]

    sell_side_levels: List[LiquidityLevel]

    active_levels: List[LiquidityLevel] = field(default_factory=list)

    swept_levels: List[LiquidityLevel] = field(default_factory=list)

    current_index: int = -1
    current_timestamp: Any = None

    error: Optional[str] = None

    @property
    def latest_sweep(self) -> Optional[LiquiditySweep]:
        return self.latest

    def confirmed_levels(
        self,
        candle_index: int,
    ) -> List[LiquidityLevel]:

        return [
            level
            for level in self.levels
            if level.confirmation_index <= candle_index
        ]

    def confirmed_sweeps(
        self,
        candle_index: int,
    ) -> List[LiquiditySweep]:

        return [
            sweep
            for sweep in self.sweeps
            if sweep.confirmation_index <= candle_index
        ]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =========================================================
# LIQUIDITY ENGINE
# =========================================================

class LiquidityEngine:
    """
    Causal liquidity engine.

    Workflow
    --------
        confirmed swings
              ↓
        liquidity levels
              ↓
        sweep detection
              ↓
        close-back confirmation
              ↓
        optional displacement confirmation
    """

    def __init__(
        self,
        swing_strength: int = DEFAULT_SWING_STRENGTH,
        lookback: int = DEFAULT_LOOKBACK,
        scan_recent: int = DEFAULT_SCAN_RECENT,
        atr_period: int = DEFAULT_ATR_PERIOD,
        equal_tolerance_atr: float = DEFAULT_EQUAL_TOLERANCE_ATR,
        min_wick_atr: float = DEFAULT_MIN_WICK_ATR,
        strong_wick_atr: float = DEFAULT_STRONG_WICK_ATR,
        volume_lookback: int = DEFAULT_VOLUME_LOOKBACK,
        volume_ratio_threshold: float = DEFAULT_VOLUME_RATIO,
        min_strength: float = DEFAULT_MIN_STRENGTH,
        require_close_back: bool = DEFAULT_REQUIRE_CLOSE_BACK,
        require_displacement: bool = DEFAULT_REQUIRE_DISPLACEMENT,
        displacement_atr: float = DEFAULT_DISPLACEMENT_ATR,
        detect_equal_levels: bool = True,
        use_volume: bool = True,
    ):

        self.swing_strength = max(int(swing_strength), 1)
        self.lookback = max(int(lookback), 20)
        self.scan_recent = max(int(scan_recent), 1)

        self.atr_period = max(int(atr_period), 1)

        self.equal_tolerance_atr = max(
            float(equal_tolerance_atr),
            0.0,
        )

        self.min_wick_atr = max(
            float(min_wick_atr),
            0.0,
        )

        self.strong_wick_atr = max(
            float(strong_wick_atr),
            self.min_wick_atr,
        )

        self.volume_lookback = max(
            int(volume_lookback),
            1,
        )

        self.volume_ratio_threshold = max(
            float(volume_ratio_threshold),
            0.0,
        )

        self.min_strength = float(min_strength)

        self.require_close_back = bool(
            require_close_back
        )

        self.require_displacement = bool(
            require_displacement
        )

        self.displacement_atr = max(
            float(displacement_atr),
            0.0,
        )

        self.detect_equal_levels = bool(
            detect_equal_levels
        )

        self.use_volume = bool(use_volume)

    # =====================================================
    # PUBLIC API
    # =====================================================

    def analyze(
        self,
        df: pd.DataFrame,
        current_index: Optional[int] = None,
        structure_result: Any = None,
    ) -> LiquidityResult:

        try:

            data = self._validate_dataframe(df)

            if data.empty:
                return self._empty_result(
                    error="Empty dataframe."
                )

            if current_index is None:
                current_index = len(data) - 1

            current_index = int(
                max(
                    0,
                    min(current_index, len(data) - 1),
                )
            )

            # ---------------------------------------------
            # CAUSAL SNAPSHOT
            # ---------------------------------------------

            snapshot = data.iloc[
                : current_index + 1
            ].copy()

            if snapshot.empty:
                return self._empty_result(
                    error="Empty causal snapshot."
                )

            # ---------------------------------------------
            # ATR
            # ---------------------------------------------

            snapshot["_atr"] = _calculate_atr(
                snapshot,
                self.atr_period,
            )

            # ---------------------------------------------
            # STRUCTURE
            # ---------------------------------------------

            if structure_result is None:

                from .structure import (
                    MarketStructureEngine,
                )

                structure_engine = (
                    MarketStructureEngine(
                        swing_strength=self.swing_strength,
                        lookback=self.lookback,
                        atr_period=self.atr_period,
                    )
                )

                structure_result = (
                    structure_engine.analyze(
                        data,
                        current_index=current_index,
                    )
                )

            # ---------------------------------------------
            # CONFIRMED SWINGS
            # ---------------------------------------------

            swings = self._extract_confirmed_swings(
                structure_result,
                current_index,
            )

            # ---------------------------------------------
            # LIQUIDITY LEVELS
            # ---------------------------------------------

            levels = self._build_liquidity_levels(
                snapshot,
                swings,
            )

            # ---------------------------------------------
            # SWEEPS
            # ---------------------------------------------

            sweeps = self._detect_sweeps(
                snapshot,
                levels,
            )

            # ---------------------------------------------
            # DISPLACEMENT
            # ---------------------------------------------

            sweeps = self._confirm_displacement(
                snapshot,
                sweeps,
            )

            # ---------------------------------------------
            # MARK SWEPT LEVELS
            # ---------------------------------------------

            self._mark_swept_levels(
                levels,
                sweeps,
            )

            # ---------------------------------------------
            # FILTER
            # ---------------------------------------------

            sweeps = [
                sweep
                for sweep in sweeps
                if sweep.strength >= self.min_strength
            ]

            sweeps.sort(
                key=lambda x: (
                    x.sweep_index,
                    x.strength,
                )
            )

            levels.sort(
                key=lambda x: (
                    x.confirmation_index,
                    x.source_index,
                )
            )

            buy_side = [
                level
                for level in levels
                if level.liquidity_side == "buy_side"
            ]

            sell_side = [
                level
                for level in levels
                if level.liquidity_side == "sell_side"
            ]

            active_levels = [
                level
                for level in levels
                if not level.swept
            ]

            swept_levels = [
                level
                for level in levels
                if level.swept
            ]

            latest = (
                sweeps[-1]
                if sweeps
                else None
            )

            return LiquidityResult(
                detected=bool(sweeps),
                latest=latest,
                sweeps=sweeps,
                levels=levels,
                buy_side_levels=buy_side,
                sell_side_levels=sell_side,
                active_levels=active_levels,
                swept_levels=swept_levels,
                current_index=current_index,
                current_timestamp=_timestamp(
                    snapshot.iloc[-1]["timestamp"]
                ),
                error=None,
            )

        except Exception as exc:

            return self._empty_result(
                error=f"{type(exc).__name__}: {exc}"
            )

    # =====================================================
    # STRUCTURE SWINGS
    # =====================================================

    def _extract_confirmed_swings(
        self,
        structure_result: Any,
        current_index: int,
    ) -> List[Any]:

        if structure_result is None:
            return []

        if hasattr(
            structure_result,
            "confirmed_swings",
        ):
            try:
                return list(
                    structure_result.confirmed_swings(
                        current_index
                    )
                )
            except Exception:
                pass

        swings = _get(
            structure_result,
            "swings",
            [],
        )

        result = []

        for swing in swings:

            confirmation_index = _safe_int(
                _get(
                    swing,
                    "confirmation_index",
                    -1,
                )
            )

            if (
                confirmation_index >= 0
                and confirmation_index <= current_index
            ):
                result.append(swing)

        return result

    # =====================================================
    # BUILD LEVELS
    # =====================================================

    def _build_liquidity_levels(
        self,
        df: pd.DataFrame,
        swings: List[Any],
    ) -> List[LiquidityLevel]:

        levels: List[LiquidityLevel] = []

        for swing in swings:

            swing_type = str(
                _get(
                    swing,
                    "swing_type",
                    "",
                )
            ).lower()

            if swing_type in {
                "high",
                "swing_high",
                "sh",
            }:

                side = "buy_side"

            elif swing_type in {
                "low",
                "swing_low",
                "sl",
            }:

                side = "sell_side"

            else:
                continue

            source_index = _safe_int(
                _get(
                    swing,
                    "pivot_index",
                    _get(swing, "index", -1),
                )
            )

            confirmation_index = _safe_int(
                _get(
                    swing,
                    "confirmation_index",
                    -1,
                )
            )

            if source_index < 0:
                continue

            if confirmation_index < 0:
                # STRICT MODE:
                # Never use an unconfirmed swing.
                continue

            if confirmation_index >= len(df):
                continue

            price = _safe_float(
                _get(swing, "price", 0.0)
            )

            if price <= 0:
                continue

            atr = _safe_float(
                df.iloc[
                    min(
                        confirmation_index,
                        len(df) - 1,
                    )
                ]["_atr"]
            )

            strength = _safe_float(
                _get(
                    swing,
                    "strength",
                    1.0,
                ),
                1.0,
            )

            level_id = (
                f"{side}_{source_index}"
            )

            levels.append(
                LiquidityLevel(
                    id=level_id,

                    liquidity_side=side,

                    liquidity_type=(
                        "swing_high"
                        if side == "buy_side"
                        else "swing_low"
                    ),

                    price=price,

                    source_index=source_index,

                    source_timestamp=_timestamp(
                        _get(
                            swing,
                            "pivot_timestamp",
                            _get(
                                swing,
                                "timestamp",
                                None,
                            ),
                        )
                    ),

                    confirmation_index=(
                        confirmation_index
                    ),

                    confirmation_timestamp=(
                        _timestamp(
                            _get(
                                swing,
                                "confirmation_timestamp",
                                None,
                            )
                        )
                    ),

                    strength=max(
                        strength,
                        1.0,
                    ),

                    atr=atr,
                )
            )

        # -------------------------------------------------
        # Equal highs / equal lows
        # -------------------------------------------------

        if self.detect_equal_levels:
            levels = self._cluster_equal_levels(
                df,
                levels,
            )

        return levels

    # =====================================================
    # EQUAL LEVEL CLUSTERING
    # =====================================================

    def _cluster_equal_levels(
        self,
        df: pd.DataFrame,
        levels: List[LiquidityLevel],
    ) -> List[LiquidityLevel]:

        if not levels:
            return []

        result: List[LiquidityLevel] = []

        groups = {
            "buy_side": [],
            "sell_side": [],
        }

        for level in levels:
            groups[
                level.liquidity_side
            ].append(level)

        for side, side_levels in groups.items():

            side_levels.sort(
                key=lambda x: (
                    x.price,
                    x.confirmation_index,
                )
            )

            used = set()

            for i, base in enumerate(side_levels):

                if i in used:
                    continue

                cluster = [base]
                used.add(i)

                for j in range(
                    i + 1,
                    len(side_levels),
                ):

                    if j in used:
                        continue

                    candidate = side_levels[j]

                    atr = max(
                        base.atr,
                        candidate.atr,
                        1e-9,
                    )

                    tolerance = (
                        atr
                        * self.equal_tolerance_atr
                    )

                    if (
                        abs(
                            candidate.price
                            - base.price
                        )
                        <= tolerance
                    ):

                        cluster.append(candidate)
                        used.add(j)

                # -----------------------------------------
                # Single swing
                # -----------------------------------------

                if len(cluster) == 1:

                    result.append(base)
                    continue

                # -----------------------------------------
                # Equal liquidity
                # -----------------------------------------

                representative = max(
                    cluster,
                    key=lambda x: (
                        x.confirmation_index,
                        x.strength,
                    )
                )

                avg_price = float(
                    np.mean(
                        [
                            item.price
                            for item in cluster
                        ]
                    )
                )

                first = min(
                    cluster,
                    key=lambda x: (
                        x.confirmation_index,
                        x.source_index,
                    )
                )

                second = (
                    cluster[1]
                    if len(cluster) > 1
                    else None
                )

                representative.price = avg_price

                representative.liquidity_type = (
                    "equal_highs"
                    if side == "buy_side"
                    else "equal_lows"
                )

                representative.equal_cluster_size = (
                    len(cluster)
                )

                representative.strength = (
                    sum(
                        max(
                            item.strength,
                            1.0,
                        )
                        for item in cluster
                    )
                    + 0.5 * (
                        len(cluster) - 1
                    )
                )

                representative.source_index = (
                    first.source_index
                )

                representative.source_timestamp = (
                    first.source_timestamp
                )

                representative.confirmation_index = max(
                    item.confirmation_index
                    for item in cluster
                )

                representative.confirmation_timestamp = (
                    _timestamp(
                        df.iloc[
                            representative.confirmation_index
                        ]["timestamp"]
                    )
                )

                if second is not None:

                    representative.source_index_2 = (
                        second.source_index
                    )

                    representative.source_timestamp_2 = (
                        second.source_timestamp
                    )

                representative.id = (
                    f"{side}_equal_"
                    f"{first.source_index}_"
                    f"{representative.confirmation_index}"
                )

                result.append(
                    representative
                )

        return result

    # =====================================================
    # SWEEP DETECTION
    # =====================================================

    def _detect_sweeps(
        self,
        df: pd.DataFrame,
        levels: List[LiquidityLevel],
    ) -> List[LiquiditySweep]:

        if df.empty or not levels:
            return []

        start_index = max(
            0,
            len(df) - self.scan_recent,
        )

        sweeps: List[LiquiditySweep] = []

        # -----------------------------------------------
        # Scan recent candles only.
        # Levels themselves come from full causal history.
        # -----------------------------------------------

        for i in range(
            start_index,
            len(df),
        ):

            candle = df.iloc[i]

            high = _safe_float(
                candle["high"]
            )

            low = _safe_float(
                candle["low"]
            )

            close = _safe_float(
                candle["close"]
            )

            open_price = _safe_float(
                candle["open"]
            )

            atr = max(
                _safe_float(
                    candle["_atr"]
                ),
                1e-9,
            )

            volume_ratio = (
                self._volume_ratio(
                    df,
                    i,
                )
            )

            # -------------------------------------------
            # Only levels that were known BEFORE sweep.
            # -------------------------------------------

            known_levels = [
                level
                for level in levels
                if (
                    level.confirmation_index < i
                    and level.source_index < i
                )
            ]

            if not known_levels:
                continue

            # -------------------------------------------
            # BUY-SIDE LIQUIDITY
            # -------------------------------------------

            buy_levels = [
                level
                for level in known_levels
                if level.liquidity_side
                == "buy_side"
            ]

            for level in buy_levels:

                # Price must actually trade above level.
                if high <= level.price:
                    continue

                wick_extension = (
                    high - level.price
                )

                wick_ratio = (
                    wick_extension / atr
                )

                if wick_ratio < self.min_wick_atr:
                    continue

                close_back = (
                    close < level.price
                )

                if (
                    self.require_close_back
                    and not close_back
                ):
                    continue

                # Bearish reaction.
                reaction = "bearish"

                strength = self._calculate_strength(
                    level=level,
                    wick_ratio=wick_ratio,
                    volume_ratio=volume_ratio,
                )

                sweeps.append(
                    LiquiditySweep(
                        id=(
                            f"sweep_"
                            f"{level.id}_"
                            f"{i}"
                        ),

                        detected=True,

                        status=(
                            "confirmed"
                            if close_back
                            else "provisional"
                        ),

                        liquidity_side="buy_side",

                        reaction=reaction,

                        liquidity_type=(
                            level.liquidity_type
                        ),

                        level=level.price,

                        extreme=high,

                        close=close,

                        wick_extension=(
                            wick_extension
                        ),

                        wick_atr_ratio=(
                            wick_ratio
                        ),

                        volume_ratio=(
                            volume_ratio
                        ),

                        strength=strength,

                        sweep_index=i,

                        sweep_timestamp=_timestamp(
                            candle["timestamp"]
                        ),

                        source_index=(
                            level.source_index
                        ),

                        source_timestamp=(
                            level.source_timestamp
                        ),

                        confirmation_index=i,

                        confirmation_timestamp=_timestamp(
                            candle["timestamp"]
                        ),

                        source_level_id=(
                            level.id
                        ),

                        close_back_confirmed=(
                            close_back
                        ),

                        displacement_confirmed=False,

                        atr=atr,
                    )
                )

            # -------------------------------------------
            # SELL-SIDE LIQUIDITY
            # -------------------------------------------

            sell_levels = [
                level
                for level in known_levels
                if level.liquidity_side
                == "sell_side"
            ]

            for level in sell_levels:

                # Price must actually trade below level.
                if low >= level.price:
                    continue

                wick_extension = (
                    level.price - low
                )

                wick_ratio = (
                    wick_extension / atr
                )

                if wick_ratio < self.min_wick_atr:
                    continue

                close_back = (
                    close > level.price
                )

                if (
                    self.require_close_back
                    and not close_back
                ):
                    continue

                # Bullish reaction.
                reaction = "bullish"

                strength = self._calculate_strength(
                    level=level,
                    wick_ratio=wick_ratio,
                    volume_ratio=volume_ratio,
                )

                sweeps.append(
                    LiquiditySweep(
                        id=(
                            f"sweep_"
                            f"{level.id}_"
                            f"{i}"
                        ),

                        detected=True,

                        status=(
                            "confirmed"
                            if close_back
                            else "provisional"
                        ),

                        liquidity_side="sell_side",

                        reaction=reaction,

                        liquidity_type=(
                            level.liquidity_type
                        ),

                        level=level.price,

                        extreme=low,

                        close=close,

                        wick_extension=(
                            wick_extension
                        ),

                        wick_atr_ratio=(
                            wick_ratio
                        ),

                        volume_ratio=(
                            volume_ratio
                        ),

                        strength=strength,

                        sweep_index=i,

                        sweep_timestamp=_timestamp(
                            candle["timestamp"]
                        ),

                        source_index=(
                            level.source_index
                        ),

                        source_timestamp=(
                            level.source_timestamp
                        ),

                        confirmation_index=i,

                        confirmation_timestamp=_timestamp(
                            candle["timestamp"]
                        ),

                        source_level_id=(
                            level.id
                        ),

                        close_back_confirmed=(
                            close_back
                        ),

                        displacement_confirmed=False,

                        atr=atr,
                    )
                )

        # -------------------------------------------------
        # Keep strongest sweep per level/candle.
        # Prevent duplicate same-level detections.
        # -------------------------------------------------

        return self._deduplicate_sweeps(
            sweeps
        )

    # =====================================================
    # SWEEP STRENGTH
    # =====================================================

    def _calculate_strength(
        self,
        level: LiquidityLevel,
        wick_ratio: float,
        volume_ratio: float,
    ) -> float:

        strength = max(
            level.strength,
            1.0,
        )

        # Strong wick
        if (
            wick_ratio
            >= self.strong_wick_atr
        ):
            strength += 2.0

        elif (
            wick_ratio
            >= self.min_wick_atr * 2
        ):
            strength += 1.0

        # Equal liquidity
        if level.equal_cluster_size >= 2:
            strength += min(
                level.equal_cluster_size,
                5,
            )

        # Volume confirmation
        if (
            self.use_volume
            and volume_ratio
            >= self.volume_ratio_threshold
        ):
            strength += 1.0

        return float(
            min(
                strength,
                20.0,
            )
        )

    # =====================================================
    # VOLUME
    # =====================================================

    def _volume_ratio(
        self,
        df: pd.DataFrame,
        index: int,
    ) -> float:

        if not self.use_volume:
            return 1.0

        if "tick_volume" not in df.columns:
            if "volume" not in df.columns:
                return 1.0

            column = "volume"

        else:
            column = "tick_volume"

        current = _safe_float(
            df.iloc[index][column]
        )

        start = max(
            0,
            index - self.volume_lookback,
        )

        previous = df.iloc[
            start:index
        ][column].astype(float)

        previous = previous[
            np.isfinite(previous)
        ]

        if previous.empty:
            return 1.0

        average = float(
            previous.mean()
        )

        if average <= 0:
            return 1.0

        return float(
            current / average
        )

    # =====================================================
    # DISPLACEMENT
    # =====================================================

    def _confirm_displacement(
        self,
        df: pd.DataFrame,
        sweeps: List[LiquiditySweep],
    ) -> List[LiquiditySweep]:

        for sweep in sweeps:

            i = sweep.sweep_index

            if i + 1 >= len(df):
                continue

            next_candle = df.iloc[
                i + 1
            ]

            next_open = _safe_float(
                next_candle["open"]
            )

            next_close = _safe_float(
                next_candle["close"]
            )

            next_high = _safe_float(
                next_candle["high"]
            )

            next_low = _safe_float(
                next_candle["low"]
            )

            atr = max(
                _safe_float(
                    next_candle["_atr"]
                ),
                1e-9,
            )

            next_range = (
                next_high
                - next_low
            )

            displacement = 0.0

            confirmed = False

            if sweep.reaction == "bearish":

                displacement = (
                    sweep.close
                    - next_close
                )

                # Next candle must move lower.
                confirmed = (
                    next_close < sweep.close
                    and displacement / atr
                    >= self.displacement_atr
                    and next_range / atr
                    >= self.displacement_atr
                )

            elif sweep.reaction == "bullish":

                displacement = (
                    next_close
                    - sweep.close
                )

                # Next candle must move higher.
                confirmed = (
                    next_close > sweep.close
                    and displacement / atr
                    >= self.displacement_atr
                    and next_range / atr
                    >= self.displacement_atr
                )

            sweep.displacement_confirmed = (
                bool(confirmed)
            )

            if confirmed:

                sweep.displacement_index = (
                    i + 1
                )

                sweep.displacement_timestamp = (
                    _timestamp(
                        next_candle[
                            "timestamp"
                        ]
                    )
                )

                sweep.displacement_atr_ratio = (
                    abs(displacement) / atr
                )

                sweep.status = (
                    "confirmed_displacement"
                )

            if (
                self.require_displacement
                and not confirmed
            ):
                sweep.status = "rejected"

        if self.require_displacement:

            sweeps = [
                sweep
                for sweep in sweeps
                if sweep.displacement_confirmed
            ]

        return sweeps

    # =====================================================
    # MARK LEVELS
    # =====================================================

    def _mark_swept_levels(
        self,
        levels: List[LiquidityLevel],
        sweeps: List[LiquiditySweep],
    ) -> None:

        for sweep in sweeps:

            for level in levels:

                if (
                    level.id
                    != sweep.source_level_id
                ):
                    continue

                # Only the first sweep marks it.
                if not level.swept:

                    level.swept = True

                    level.swept_index = (
                        sweep.sweep_index
                    )

                    level.swept_timestamp = (
                        sweep.sweep_timestamp
                    )

                break

    # =====================================================
    # DEDUPLICATION
    # =====================================================

    def _deduplicate_sweeps(
        self,
        sweeps: List[LiquiditySweep],
    ) -> List[LiquiditySweep]:

        if not sweeps:
            return []

        grouped: Dict[
            Tuple[str, int],
            LiquiditySweep,
        ] = {}

        for sweep in sweeps:

            key = (
                sweep.source_level_id,
                sweep.sweep_index,
            )

            previous = grouped.get(key)

            if (
                previous is None
                or sweep.strength
                > previous.strength
            ):
                grouped[key] = sweep

        result = list(
            grouped.values()
        )

        result.sort(
            key=lambda x: (
                x.sweep_index,
                x.strength,
            )
        )

        return result

    # =====================================================
    # VALIDATION
    # =====================================================

    def _validate_dataframe(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        if df is None:
            raise ValueError(
                "DataFrame is None."
            )

        if not isinstance(
            df,
            pd.DataFrame,
        ):
            raise TypeError(
                "df must be pandas.DataFrame."
            )

        if df.empty:
            return pd.DataFrame(
                columns=[
                    "open",
                    "high",
                    "low",
                    "close",
                    "timestamp",
                ]
            )

        data = df.copy()

        # ---------------------------------------------
        # Column aliases
        # ---------------------------------------------

        rename_map = {}

        for column in data.columns:

            normalized = (
                str(column)
                .strip()
                .lower()
                .replace(" ", "_")
            )

            if normalized in {
                "open",
                "o",
            }:
                rename_map[column] = "open"

            elif normalized in {
                "high",
                "h",
            }:
                rename_map[column] = "high"

            elif normalized in {
                "low",
                "l",
            }:
                rename_map[column] = "low"

            elif normalized in {
                "close",
                "c",
            }:
                rename_map[column] = "close"

            elif normalized in {
                "tick_volume",
                "tickvolume",
            }:
                rename_map[column] = "tick_volume"

            elif normalized in {
                "volume",
                "real_volume",
            }:
                rename_map[column] = "volume"

            elif normalized in {
                "timestamp",
                "time",
                "datetime",
                "date",
            }:
                rename_map[column] = "timestamp"

        data = data.rename(
            columns=rename_map
        )

        required = [
            "open",
            "high",
            "low",
            "close",
        ]

        missing = [
            column
            for column in required
            if column not in data.columns
        ]

        if missing:
            raise ValueError(
                "Missing OHLC columns: "
                + ", ".join(missing)
            )

        # ---------------------------------------------
        # Timestamp
        # ---------------------------------------------

        if "timestamp" not in data.columns:

            data["timestamp"] = pd.RangeIndex(
                start=0,
                stop=len(data),
            )

        else:

            parsed = pd.to_datetime(
                data["timestamp"],
                errors="coerce",
            )

            data["timestamp"] = parsed

            data = data[
                data["timestamp"].notna()
            ]

        # ---------------------------------------------
        # Numeric OHLC
        # ---------------------------------------------

        for column in required:

            data[column] = pd.to_numeric(
                data[column],
                errors="coerce",
            )

        data = data.dropna(
            subset=required
        )

        # ---------------------------------------------
        # Remove invalid candles
        # ---------------------------------------------

        data = data[
            (data["high"] >= data["low"])
            & (data["high"] >= data["open"])
            & (data["high"] >= data["close"])
            & (data["low"] <= data["open"])
            & (data["low"] <= data["close"])
        ]

        # ---------------------------------------------
        # Chronological order
        # ---------------------------------------------

        data = (
            data.sort_values(
                "timestamp",
                kind="stable",
            )
            .reset_index(drop=True)
        )

        return data

    # =====================================================
    # EMPTY RESULT
    # =====================================================

    def _empty_result(
        self,
        error: Optional[str] = None,
    ) -> LiquidityResult:

        return LiquidityResult(
            detected=False,
            latest=None,
            sweeps=[],
            levels=[],
            buy_side_levels=[],
            sell_side_levels=[],
            active_levels=[],
            swept_levels=[],
            current_index=-1,
            current_timestamp=None,
            error=error,
        )


# =========================================================
# CONVENIENCE FUNCTIONS
# =========================================================

def detect_liquidity(
    df: pd.DataFrame,
    current_index: Optional[int] = None,
    **kwargs,
) -> LiquidityResult:

    engine = LiquidityEngine(
        **kwargs
    )

    return engine.analyze(
        df,
        current_index=current_index,
    )


def analyze_liquidity(
    df: pd.DataFrame,
    current_index: Optional[int] = None,
    **kwargs,
) -> LiquidityResult:

    return detect_liquidity(
        df,
        current_index=current_index,
        **kwargs,
    )


# =========================================================
# COMPATIBILITY ALIASES
# =========================================================

LiquidityEngineV2 = LiquidityEngine
LiquidityEngineV21 = LiquidityEngine


__all__ = [
    "LiquidityLevel",
    "LiquiditySweep",
    "LiquidityResult",
    "LiquidityEngine",
    "LiquidityEngineV2",
    "LiquidityEngineV21",
    "detect_liquidity",
    "analyze_liquidity",
]