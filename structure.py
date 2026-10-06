"""
smc/structure.py
===========================================================
Market Structure Engine V2.1 - FIXED INDEX / CONFIRMATION
===========================================================

Core principles
---------------
1. All indexes are GLOBAL POSITIONAL indexes.
2. pivot_index != confirmation_index.
3. A swing is usable only after confirmation.
4. A structure event is usable only after its confirmation.
5. No future candle is used.
6. Lookback slicing NEVER changes index identity.
7. Works candle-by-candle for backtesting.
8. Compatible with Liquidity / FVG / OB / Zones / SMC Engine.

Index contract
--------------
pivot_index:
    Candle where the actual swing occurred.

confirmation_index:
    First candle where the swing is objectively confirmed.

event.index:
    Candle where BOS/MSS/CHOCH actually occurs.

event.confirmation_index:
    First candle where the event is known.
    For close-confirmed breaks this is normally event.index.

current_index:
    Current GLOBAL positional candle being analyzed.

Example
-------
Swing High at candle 100 with strength=5:

    pivot_index       = 100
    confirmation_index = 105

At candle 102:
    swing is NOT known.

At candle 105:
    swing becomes known.

At candle 110:
    it may be used for BOS/liquidity/etc.
===========================================================
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional
import math

import numpy as np
import pandas as pd


# =========================================================
# DEFAULTS
# =========================================================

DEFAULT_SWING_STRENGTH = 5
DEFAULT_LOOKBACK = 200
DEFAULT_BREAK_LOOKBACK = 50

DEFAULT_ATR_PERIOD = 14
DEFAULT_MIN_DISPLACEMENT_ATR = 0.0

DEFAULT_REQUIRE_CLOSE_BREAK = True

DEFAULT_ALLOW_EQUAL_SWINGS = False
DEFAULT_EQUAL_TOLERANCE_ATR = 0.05


# =========================================================
# HELPERS
# =========================================================

def _get(
    obj: Any,
    name: str,
    default: Any = None,
) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)

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

        if value is None:
            return default

        result = float(value)

        if not math.isfinite(result):
            return default

        return result

    except (TypeError, ValueError):
        return default


def _safe_int(
    value: Any,
    default: int = -1,
) -> int:

    try:
        return int(value)

    except (TypeError, ValueError):
        return default


def _timestamp(
    value: Any,
) -> Any:

    if value is None:
        return None

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    try:
        return pd.Timestamp(value).isoformat()

    except Exception:
        return str(value)


def _normalize_direction(
    value: Any,
) -> Optional[str]:

    if value is None:
        return None

    value = str(value).strip().lower()

    mapping = {
        "bull": "bullish",
        "bullish": "bullish",
        "buy": "bullish",
        "long": "bullish",

        "bear": "bearish",
        "bearish": "bearish",
        "sell": "bearish",
        "short": "bearish",
    }

    return mapping.get(value)


# =========================================================
# ATR
# =========================================================

def _calculate_atr(
    df: pd.DataFrame,
    period: int = DEFAULT_ATR_PERIOD,
) -> pd.Series:

    high = df["high"]
    low = df["low"]
    close = df["close"]

    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return true_range.ewm(
        alpha=1.0 / max(period, 1),
        adjust=False,
        min_periods=1,
    ).mean()


# =========================================================
# DATACLASSES
# =========================================================

@dataclass
class SwingPoint:
    """
    Confirmed market swing.

    IMPORTANT:
        pivot_index is NOT the same as confirmation_index.
    """

    id: str

    swing_type: str

    price: float

    pivot_index: int
    pivot_timestamp: Any

    confirmation_index: int
    confirmation_timestamp: Any

    strength: int

    atr: float = 0.0

    classification: Optional[str] = None

    @property
    def index(self) -> int:
        """Backward-compatible alias."""
        return self.pivot_index

    @property
    def timestamp(self) -> Any:
        """Backward-compatible alias."""
        return self.pivot_timestamp

    @property
    def confirmed(self) -> bool:
        return self.confirmation_index >= 0


@dataclass
class StructureEvent:
    """
    BOS / MSS / CHOCH event.

    event.index:
        actual break candle.

    confirmation_index:
        candle where the event became known.
    """

    id: str

    event_type: str

    direction: str

    index: int
    timestamp: Any

    confirmation_index: int
    confirmation_timestamp: Any

    price: float

    broken_swing_id: Optional[str] = None

    broken_swing_index: Optional[int] = None

    broken_swing_price: Optional[float] = None

    previous_structure: Optional[str] = None

    displacement: float = 0.0

    displacement_atr_ratio: float = 0.0

    score: float = 0.0

    @property
    def type(self) -> str:
        """Backward-compatible alias."""
        return self.event_type

    @property
    def confirmed(self) -> bool:
        return self.confirmation_index >= 0


@dataclass
class StructureResult:
    detected: bool

    trend_bias: str

    structure_state: str

    current_index: int
    current_timestamp: Any

    swing_highs: List[SwingPoint] = field(
        default_factory=list
    )

    swing_lows: List[SwingPoint] = field(
        default_factory=list
    )

    swings: List[SwingPoint] = field(
        default_factory=list
    )

    events: List[StructureEvent] = field(
        default_factory=list
    )

    bos_events: List[StructureEvent] = field(
        default_factory=list
    )

    mss_events: List[StructureEvent] = field(
        default_factory=list
    )

    choch_events: List[StructureEvent] = field(
        default_factory=list
    )

    latest_swing_high: Optional[SwingPoint] = None

    latest_swing_low: Optional[SwingPoint] = None

    latest_event: Optional[StructureEvent] = None

    latest_bos: Optional[StructureEvent] = None

    latest_mss: Optional[StructureEvent] = None

    latest_choch: Optional[StructureEvent] = None

    error: Optional[str] = None

    # -----------------------------------------------------
    # Causal filters
    # -----------------------------------------------------

    def confirmed_swings(
        self,
        candle_index: int,
    ) -> List[SwingPoint]:

        return [
            swing
            for swing in self.swings
            if (
                swing.confirmation_index >= 0
                and swing.confirmation_index
                <= candle_index
            )
        ]

    def confirmed_events(
        self,
        candle_index: int,
    ) -> List[StructureEvent]:

        return [
            event
            for event in self.events
            if (
                event.confirmation_index >= 0
                and event.confirmation_index
                <= candle_index
            )
        ]

    def to_dict(
        self,
    ) -> Dict[str, Any]:

        return asdict(self)


# =========================================================
# MARKET STRUCTURE ENGINE
# =========================================================

class MarketStructureEngine:
    """
    Causal market structure detector.

    The engine always works with GLOBAL positional indexes.

    It is safe to call:

        analyze(df, current_index=i)

    repeatedly in a candle-by-candle backtest.
    """

    def __init__(
        self,
        swing_strength: int = DEFAULT_SWING_STRENGTH,
        lookback: int = DEFAULT_LOOKBACK,
        break_lookback: int = DEFAULT_BREAK_LOOKBACK,
        min_displacement_atr: float = (
            DEFAULT_MIN_DISPLACEMENT_ATR
        ),
        atr_period: int = DEFAULT_ATR_PERIOD,
        require_close_break: bool = (
            DEFAULT_REQUIRE_CLOSE_BREAK
        ),
        allow_equal_swings: bool = (
            DEFAULT_ALLOW_EQUAL_SWINGS
        ),
        equal_tolerance_atr: float = (
            DEFAULT_EQUAL_TOLERANCE_ATR
        ),
    ):

        self.swing_strength = max(
            int(swing_strength),
            1,
        )

        self.lookback = max(
            int(lookback),
            20,
        )

        self.break_lookback = max(
            int(break_lookback),
            5,
        )

        self.min_displacement_atr = max(
            float(min_displacement_atr),
            0.0,
        )

        self.atr_period = max(
            int(atr_period),
            1,
        )

        self.require_close_break = bool(
            require_close_break
        )

        self.allow_equal_swings = bool(
            allow_equal_swings
        )

        self.equal_tolerance_atr = max(
            float(equal_tolerance_atr),
            0.0,
        )

    # =====================================================
    # PUBLIC ANALYZE
    # =====================================================

    def analyze(
        self,
        df: pd.DataFrame,
        current_index: Optional[int] = None,
    ) -> StructureResult:

        try:

            data = self._validate_dataframe(df)

            if data.empty:

                return self._empty_result(
                    error="Empty dataframe."
                )

            # -------------------------------------------------
            # CURRENT INDEX IS GLOBAL POSITIONAL INDEX
            # -------------------------------------------------

            if current_index is None:

                current_index = len(data) - 1

            current_index = int(
                max(
                    0,
                    min(
                        current_index,
                        len(data) - 1,
                    ),
                )
            )

            # -------------------------------------------------
            # CAUSAL SNAPSHOT
            #
            # Nothing after current_index exists here.
            # -------------------------------------------------

            snapshot = data.iloc[
                : current_index + 1
            ].copy()

            if snapshot.empty:

                return self._empty_result(
                    error="Empty causal snapshot."
                )

            snapshot["_atr"] = _calculate_atr(
                snapshot,
                self.atr_period,
            )

            # -------------------------------------------------
            # GLOBAL LOOKBACK
            #
            # IMPORTANT:
            # Do NOT reset index.
            #
            # snapshot index:
            #     0,1,2,...,current_index
            #
            # work index:
            #     start,...,current_index
            #
            # But all calculations below explicitly convert
            # local position -> GLOBAL index.
            # -------------------------------------------------

            start = max(
                0,
                len(snapshot) - self.lookback,
            )

            work = snapshot.iloc[
                start:
            ].copy()

            # Explicit global index column.
            work["_global_index"] = np.arange(
                start,
                current_index + 1,
            )

            # -------------------------------------------------
            # SWINGS
            # -------------------------------------------------

            swings = self.find_swing_points(
                work,
                current_index=current_index,
                global_index_column="_global_index",
            )

            # -------------------------------------------------
            # CLASSIFY HH / HL / LH / LL
            # -------------------------------------------------

            self._classify_swings(
                swings
            )

            # -------------------------------------------------
            # STRUCTURE EVENTS
            # -------------------------------------------------

            events = self._detect_structure_events(
                work,
                swings,
                current_index=current_index,
                global_index_column="_global_index",
            )

            # -------------------------------------------------
            # CHOCH
            # -------------------------------------------------

            events = self._detect_choch(
                events
            )

            # -------------------------------------------------
            # SORT
            # -------------------------------------------------

            swings.sort(
                key=lambda x: (
                    x.confirmation_index,
                    x.pivot_index,
                )
            )

            events.sort(
                key=lambda x: (
                    x.confirmation_index,
                    x.index,
                )
            )

            # -------------------------------------------------
            # FILTER CONFIRMED AT CURRENT CANDLE
            # -------------------------------------------------

            confirmed_swings = [
                swing
                for swing in swings
                if (
                    swing.confirmation_index
                    <= current_index
                )
            ]

            confirmed_events = [
                event
                for event in events
                if (
                    event.confirmation_index
                    <= current_index
                )
            ]

            # -------------------------------------------------
            # TREND / STATE
            # -------------------------------------------------

            trend_bias = (
                self._infer_trend_bias(
                    confirmed_events,
                    confirmed_swings,
                )
            )

            structure_state = (
                self._infer_structure_state(
                    confirmed_events
                )
            )

            # -------------------------------------------------
            # SPLIT EVENTS
            # -------------------------------------------------

            bos_events = [
                event
                for event in confirmed_events
                if event.event_type == "BOS"
            ]

            mss_events = [
                event
                for event in confirmed_events
                if event.event_type == "MSS"
            ]

            choch_events = [
                event
                for event in confirmed_events
                if event.event_type == "CHOCH"
            ]

            swing_highs = [
                swing
                for swing in confirmed_swings
                if swing.swing_type == "high"
            ]

            swing_lows = [
                swing
                for swing in confirmed_swings
                if swing.swing_type == "low"
            ]

            return StructureResult(

                detected=bool(
                    confirmed_swings
                    or confirmed_events
                ),

                trend_bias=trend_bias,

                structure_state=structure_state,

                current_index=current_index,

                current_timestamp=_timestamp(
                    snapshot.iloc[-1]["timestamp"]
                ),

                swing_highs=swing_highs,

                swing_lows=swing_lows,

                swings=confirmed_swings,

                events=confirmed_events,

                bos_events=bos_events,

                mss_events=mss_events,

                choch_events=choch_events,

                latest_swing_high=(
                    swing_highs[-1]
                    if swing_highs
                    else None
                ),

                latest_swing_low=(
                    swing_lows[-1]
                    if swing_lows
                    else None
                ),

                latest_event=(
                    confirmed_events[-1]
                    if confirmed_events
                    else None
                ),

                latest_bos=(
                    bos_events[-1]
                    if bos_events
                    else None
                ),

                latest_mss=(
                    mss_events[-1]
                    if mss_events
                    else None
                ),

                latest_choch=(
                    choch_events[-1]
                    if choch_events
                    else None
                ),

                error=None,
            )

        except Exception as exc:

            return self._empty_result(
                error=(
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )
            )

    # =====================================================
    # SWING DETECTION
    # =====================================================

    def find_swing_points(
        self,
        df: pd.DataFrame,
        current_index: Optional[int] = None,
        global_index_column: Optional[str] = None,
    ) -> List[SwingPoint]:

        if df.empty:
            return []

        if current_index is None:
            current_index = len(df) - 1

        current_index = int(
            current_index
        )

        strength = self.swing_strength

        swings: List[SwingPoint] = []

        # -------------------------------------------------
        # Local position loop.
        #
        # i is LOCAL position in work.
        # global_i is GLOBAL position in original df.
        # -------------------------------------------------

        for i in range(
            strength,
            len(df) - strength,
        ):

            if (
                global_index_column
                and global_index_column in df.columns
            ):

                global_i = _safe_int(
                    df.iloc[i][
                        global_index_column
                    ],
                    i,
                )

            else:

                # If dataframe itself is the full/global
                # dataframe, iloc position == global index.
                global_i = i

            confirmation_global = (
                global_i + strength
            )

            # ---------------------------------------------
            # Causal confirmation check
            # ---------------------------------------------

            if (
                confirmation_global
                > current_index
            ):
                continue

            pivot_high = _safe_float(
                df.iloc[i]["high"]
            )

            pivot_low = _safe_float(
                df.iloc[i]["low"]
            )

            atr = _safe_float(
                df.iloc[i]["_atr"]
                if "_atr" in df.columns
                else 0.0
            )

            # ---------------------------------------------
            # LEFT / RIGHT WINDOWS
            # ---------------------------------------------

            left = df.iloc[
                i - strength : i
            ]

            right = df.iloc[
                i + 1 : i + strength + 1
            ]

            left_high = (
                left["high"].max()
            )

            right_high = (
                right["high"].max()
            )

            left_low = (
                left["low"].min()
            )

            right_low = (
                right["low"].min()
            )

            # ---------------------------------------------
            # SWING HIGH
            # ---------------------------------------------

            is_swing_high = (
                pivot_high >= left_high
                and pivot_high >= right_high
            )

            # ---------------------------------------------
            # SWING LOW
            # ---------------------------------------------

            is_swing_low = (
                pivot_low <= left_low
                and pivot_low <= right_low
            )

            # ---------------------------------------------
            # Equal swing policy
            # ---------------------------------------------

            if not self.allow_equal_swings:

                tolerance = (
                    max(atr, 1e-9)
                    * self.equal_tolerance_atr
                )

                # If pivot is essentially equal to
                # another surrounding extreme, don't
                # create ambiguous duplicate swing.
                #
                # We only suppress if BOTH sides contain
                # an equivalent extreme.
                if is_swing_high:

                    equal_high_count = (
                        int(
                            (
                                (
                                    left["high"]
                                    - pivot_high
                                ).abs()
                                <= tolerance
                            ).sum()
                        )
                        +
                        int(
                            (
                                (
                                    right["high"]
                                    - pivot_high
                                ).abs()
                                <= tolerance
                            ).sum()
                        )
                    )

                    if equal_high_count > 0:
                        is_swing_high = False

                if is_swing_low:

                    equal_low_count = (
                        int(
                            (
                                (
                                    left["low"]
                                    - pivot_low
                                ).abs()
                                <= tolerance
                            ).sum()
                        )
                        +
                        int(
                            (
                                (
                                    right["low"]
                                    - pivot_low
                                ).abs()
                                <= tolerance
                            ).sum()
                        )
                    )

                    if equal_low_count > 0:
                        is_swing_low = False

            # -------------------------------------------------
            # BOTH HIGH AND LOW
            #
            # A candle can theoretically be both.
            # We avoid creating ambiguous structure.
            # -------------------------------------------------

            if (
                is_swing_high
                and is_swing_low
            ):

                candle_range = (
                    pivot_high
                    - pivot_low
                )

                if (
                    candle_range
                    >= max(atr, 1e-9)
                ):
                    # Ignore ambiguous giant candle.
                    is_swing_high = False
                    is_swing_low = False

            # -------------------------------------------------
            # CREATE HIGH
            # -------------------------------------------------

            if is_swing_high:

                swings.append(
                    SwingPoint(

                        id=(
                            f"SH_{global_i}"
                        ),

                        swing_type="high",

                        price=pivot_high,

                        pivot_index=global_i,

                        pivot_timestamp=_timestamp(
                            df.iloc[i][
                                "timestamp"
                            ]
                        ),

                        confirmation_index=(
                            confirmation_global
                        ),

                        confirmation_timestamp=_timestamp(
                            df.iloc[
                                i + strength
                            ][
                                "timestamp"
                            ]
                        ),

                        strength=strength,

                        atr=atr,
                    )
                )

            # -------------------------------------------------
            # CREATE LOW
            # -------------------------------------------------

            if is_swing_low:

                swings.append(
                    SwingPoint(

                        id=(
                            f"SL_{global_i}"
                        ),

                        swing_type="low",

                        price=pivot_low,

                        pivot_index=global_i,

                        pivot_timestamp=_timestamp(
                            df.iloc[i][
                                "timestamp"
                            ]
                        ),

                        confirmation_index=(
                            confirmation_global
                        ),

                        confirmation_timestamp=_timestamp(
                            df.iloc[
                                i + strength
                            ][
                                "timestamp"
                            ]
                        ),

                        strength=strength,

                        atr=atr,
                    )
                )

        return swings

    # =====================================================
    # CLASSIFY SWINGS
    # =====================================================

    def _classify_swings(
        self,
        swings: List[SwingPoint],
    ) -> None:

        highs = sorted(
            [
                swing
                for swing in swings
                if swing.swing_type == "high"
            ],
            key=lambda x: (
                x.confirmation_index,
                x.pivot_index,
            ),
        )

        lows = sorted(
            [
                swing
                for swing in swings
                if swing.swing_type == "low"
            ],
            key=lambda x: (
                x.confirmation_index,
                x.pivot_index,
            ),
        )

        previous_high: Optional[
            SwingPoint
        ] = None

        for swing in highs:

            if previous_high is None:

                swing.classification = "H"

            else:

                if swing.price > previous_high.price:
                    swing.classification = "HH"

                elif swing.price < previous_high.price:
                    swing.classification = "LH"

                else:
                    swing.classification = "EH"

            previous_high = swing

        previous_low: Optional[
            SwingPoint
        ] = None

        for swing in lows:

            if previous_low is None:

                swing.classification = "L"

            else:

                if swing.price > previous_low.price:
                    swing.classification = "HL"

                elif swing.price < previous_low.price:
                    swing.classification = "LL"

                else:
                    swing.classification = "EL"

            previous_low = swing

    # =====================================================
    # STRUCTURE EVENTS
    # =====================================================

    def _detect_structure_events(
        self,
        df: pd.DataFrame,
        swings: List[SwingPoint],
        current_index: int,
        global_index_column: str = "_global_index",
    ) -> List[StructureEvent]:

        events: List[StructureEvent] = []

        if df.empty or not swings:
            return events

        confirmed_highs = sorted(
            [
                swing
                for swing in swings
                if swing.swing_type == "high"
            ],
            key=lambda x: (
                x.confirmation_index,
                x.pivot_index,
            ),
        )

        confirmed_lows = sorted(
            [
                swing
                for swing in swings
                if swing.swing_type == "low"
            ],
            key=lambda x: (
                x.confirmation_index,
                x.pivot_index,
            ),
        )

        # -------------------------------------------------
        # Track which swing levels have already been broken.
        #
        # This prevents:
        #
        # close > same high
        # close > same high
        # close > same high
        #
        # from generating 3 BOS events.
        # -------------------------------------------------

        broken_high_ids = set()
        broken_low_ids = set()

        # -------------------------------------------------
        # Global candle loop
        # -------------------------------------------------

        for local_i in range(
            len(df)
        ):

            global_i = _safe_int(
                df.iloc[local_i][
                    global_index_column
                ],
                local_i,
            )

            if global_i > current_index:
                continue

            # -------------------------------------------------
            # We need a confirmed swing BEFORE the break.
            # -------------------------------------------------

            available_highs = [
                swing
                for swing in confirmed_highs
                if (
                    swing.confirmation_index
                    < global_i
                    and swing.pivot_index
                    < global_i
                )
            ]

            available_lows = [
                swing
                for swing in confirmed_lows
                if (
                    swing.confirmation_index
                    < global_i
                    and swing.pivot_index
                    < global_i
                )
            ]

            if not available_highs and not available_lows:
                continue

            candle = df.iloc[local_i]

            close = _safe_float(
                candle["close"]
            )

            open_price = _safe_float(
                candle["open"]
            )

            high = _safe_float(
                candle["high"]
            )

            low = _safe_float(
                candle["low"]
            )

            atr = max(
                _safe_float(
                    candle["_atr"]
                    if "_atr" in df.columns
                    else 0.0
                ),
                1e-9,
            )

            # -------------------------------------------------
            # Break lookback
            #
            # Only consider recent structure levels.
            # But their GLOBAL identity remains intact.
            # -------------------------------------------------

            min_break_index = (
                global_i
                - self.break_lookback
            )

            available_highs = [
                swing
                for swing in available_highs
                if (
                    swing.confirmation_index
                    >= min_break_index
                )
            ]

            available_lows = [
                swing
                for swing in available_lows
                if (
                    swing.confirmation_index
                    >= min_break_index
                )
            ]

            # -------------------------------------------------
            # Most recent confirmed HIGH / LOW
            # -------------------------------------------------

            last_high = (
                max(
                    available_highs,
                    key=lambda x: (
                        x.confirmation_index,
                        x.pivot_index,
                    ),
                )
                if available_highs
                else None
            )

            last_low = (
                max(
                    available_lows,
                    key=lambda x: (
                        x.confirmation_index,
                        x.pivot_index,
                    ),
                )
                if available_lows
                else None
            )

            # -------------------------------------------------
            # BULLISH BREAK
            # -------------------------------------------------

            bullish_break = False

            if last_high is not None:

                if (
                    self.require_close_break
                ):

                    bullish_break = (
                        close
                        > last_high.price
                    )

                else:

                    bullish_break = (
                        high
                        > last_high.price
                    )

            # -------------------------------------------------
            # BEARISH BREAK
            # -------------------------------------------------

            bearish_break = False

            if last_low is not None:

                if (
                    self.require_close_break
                ):

                    bearish_break = (
                        close
                        < last_low.price
                    )

                else:

                    bearish_break = (
                        low
                        < last_low.price
                    )

            # -------------------------------------------------
            # Ignore simultaneous ambiguous break.
            # -------------------------------------------------

            if (
                bullish_break
                and bearish_break
            ):

                candle_body = (
                    close - open_price
                )

                if candle_body > 0:

                    bearish_break = False

                elif candle_body < 0:

                    bullish_break = False

                else:

                    bullish_break = False
                    bearish_break = False

            # -------------------------------------------------
            # DISPLACEMENT
            # -------------------------------------------------

            bullish_displacement = (
                close
                - last_high.price
                if last_high is not None
                else 0.0
            )

            bearish_displacement = (
                last_low.price
                - close
                if last_low is not None
                else 0.0
            )

            # -------------------------------------------------
            # Minimum displacement
            # -------------------------------------------------

            if bullish_break:

                ratio = (
                    bullish_displacement
                    / atr
                )

                if (
                    ratio
                    < self.min_displacement_atr
                ):
                    bullish_break = False

            if bearish_break:

                ratio = (
                    bearish_displacement
                    / atr
                )

                if (
                    ratio
                    < self.min_displacement_atr
                ):
                    bearish_break = False

            # -------------------------------------------------
            # CREATE BULLISH EVENT
            # -------------------------------------------------

            if (
                bullish_break
                and last_high is not None
                and last_high.id
                not in broken_high_ids
            ):

                event = StructureEvent(

                    id=(
                        f"BOS_RAW_BULL_"
                        f"{global_i}_"
                        f"{last_high.id}"
                    ),

                    event_type="BOS",

                    direction="bullish",

                    index=global_i,

                    timestamp=_timestamp(
                        candle["timestamp"]
                    ),

                    confirmation_index=global_i,

                    confirmation_timestamp=_timestamp(
                        candle["timestamp"]
                    ),

                    price=close,

                    broken_swing_id=(
                        last_high.id
                    ),

                    broken_swing_index=(
                        last_high.pivot_index
                    ),

                    broken_swing_price=(
                        last_high.price
                    ),

                    previous_structure=(
                        last_high.classification
                    ),

                    displacement=(
                        bullish_displacement
                    ),

                    displacement_atr_ratio=(
                        bullish_displacement
                        / atr
                    ),

                    score=self._event_score(
                        "BOS",
                        bullish_displacement / atr,
                    ),
                )

                events.append(event)

                broken_high_ids.add(
                    last_high.id
                )

            # -------------------------------------------------
            # CREATE BEARISH EVENT
            # -------------------------------------------------

            if (
                bearish_break
                and last_low is not None
                and last_low.id
                not in broken_low_ids
            ):

                event = StructureEvent(

                    id=(
                        f"BOS_RAW_BEAR_"
                        f"{global_i}_"
                        f"{last_low.id}"
                    ),

                    event_type="BOS",

                    direction="bearish",

                    index=global_i,

                    timestamp=_timestamp(
                        candle["timestamp"]
                    ),

                    confirmation_index=global_i,

                    confirmation_timestamp=_timestamp(
                        candle["timestamp"]
                    ),

                    price=close,

                    broken_swing_id=(
                        last_low.id
                    ),

                    broken_swing_index=(
                        last_low.pivot_index
                    ),

                    broken_swing_price=(
                        last_low.price
                    ),

                    previous_structure=(
                        last_low.classification
                    ),

                    displacement=(
                        bearish_displacement
                    ),

                    displacement_atr_ratio=(
                        bearish_displacement
                        / atr
                    ),

                    score=self._event_score(
                        "BOS",
                        bearish_displacement / atr,
                    ),
                )

                events.append(event)

                broken_low_ids.add(
                    last_low.id
                )

        # -------------------------------------------------
        # Convert BOS/MSS according to chronological state
        # -------------------------------------------------

        return self._classify_bos_mss(
            events
        )

    # =====================================================
    # BOS / MSS
    # =====================================================

    def _classify_bos_mss(
        self,
        events: List[StructureEvent],
    ) -> List[StructureEvent]:

        if not events:
            return []

        events = sorted(
            events,
            key=lambda x: (
                x.confirmation_index,
                x.index,
            ),
        )

        structural_bias: Optional[
            str
        ] = None

        result: List[
            StructureEvent
        ] = []

        for event in events:

            if structural_bias is None:

                event.event_type = "BOS"

                structural_bias = (
                    event.direction
                )

            elif (
                event.direction
                == structural_bias
            ):

                event.event_type = "BOS"

            else:

                event.event_type = "MSS"

                structural_bias = (
                    event.direction
                )

            event.score = self._event_score(
                event.event_type,
                event.displacement_atr_ratio,
            )

            result.append(event)

        return result

    # =====================================================
    # CHOCH
    # =====================================================

    def _detect_choch(
        self,
        events: List[StructureEvent],
    ) -> List[StructureEvent]:

        if not events:
            return []

        ordered = sorted(
            events,
            key=lambda x: (
                x.confirmation_index,
                x.index,
            ),
        )

        result: List[
            StructureEvent
        ] = []

        previous_direction: Optional[
            str
        ] = None

        for event in ordered:

            # -------------------------------------------------
            # MSS is the structural transition.
            #
            # We also expose CHOCH as a separate semantic
            # event when the confirmed break reverses the
            # previously established direction.
            # -------------------------------------------------

            if (
                event.event_type == "MSS"
                and previous_direction is not None
                and event.direction
                != previous_direction
            ):

                choch = StructureEvent(

                    id=(
                        f"CHOCH_"
                        f"{event.index}"
                    ),

                    event_type="CHOCH",

                    direction=event.direction,

                    index=event.index,

                    timestamp=event.timestamp,

                    confirmation_index=(
                        event.confirmation_index
                    ),

                    confirmation_timestamp=(
                        event.confirmation_timestamp
                    ),

                    price=event.price,

                    broken_swing_id=(
                        event.broken_swing_id
                    ),

                    broken_swing_index=(
                        event.broken_swing_index
                    ),

                    broken_swing_price=(
                        event.broken_swing_price
                    ),

                    previous_structure=(
                        event.previous_structure
                    ),

                    displacement=(
                        event.displacement
                    ),

                    displacement_atr_ratio=(
                        event.displacement_atr_ratio
                    ),

                    score=self._event_score(
                        "CHOCH",
                        event.displacement_atr_ratio,
                    ),
                )

                result.append(choch)

            result.append(event)

            previous_direction = (
                event.direction
            )

        return result

    # =====================================================
    # EVENT SCORE
    # =====================================================

    def _event_score(
        self,
        event_type: str,
        displacement_ratio: float,
    ) -> float:

        base = {
            "BOS": 60.0,
            "MSS": 75.0,
            "CHOCH": 90.0,
        }.get(
            event_type,
            50.0,
        )

        displacement_bonus = min(
            max(
                displacement_ratio,
                0.0,
            )
            * 10.0,
            10.0,
        )

        return min(
            base + displacement_bonus,
            100.0,
        )

    # =====================================================
    # TREND BIAS
    # =====================================================

    def _infer_trend_bias(
        self,
        events: List[StructureEvent],
        swings: List[SwingPoint],
    ) -> str:

        if events:

            latest = max(
                events,
                key=lambda x: (
                    x.confirmation_index,
                    x.index,
                ),
            )

            return latest.direction

        # ---------------------------------------------
        # Fallback to swing structure.
        # ---------------------------------------------

        highs = [
            swing
            for swing in swings
            if swing.swing_type == "high"
        ]

        lows = [
            swing
            for swing in swings
            if swing.swing_type == "low"
        ]

        bullish = 0
        bearish = 0

        for swing in highs:

            if swing.classification == "HH":
                bullish += 1

            elif swing.classification == "LH":
                bearish += 1

        for swing in lows:

            if swing.classification == "HL":
                bullish += 1

            elif swing.classification == "LL":
                bearish += 1

        if bullish > bearish:
            return "bullish"

        if bearish > bullish:
            return "bearish"

        return "neutral"

    # =====================================================
    # STRUCTURE STATE
    # =====================================================

    def _infer_structure_state(
        self,
        events: List[StructureEvent],
    ) -> str:

        if not events:
            return "undefined"

        latest = max(
            events,
            key=lambda x: (
                x.confirmation_index,
                x.index,
            ),
        )

        return (
            f"{latest.event_type}_"
            f"{latest.direction}"
        )

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

        # -------------------------------------------------
        # Normalize column names
        # -------------------------------------------------

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

        # -------------------------------------------------
        # Timestamp
        # -------------------------------------------------

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

        # -------------------------------------------------
        # Numeric OHLC
        # -------------------------------------------------

        for column in required:

            data[column] = pd.to_numeric(
                data[column],
                errors="coerce",
            )

        data = data.dropna(
            subset=required
        )

        # -------------------------------------------------
        # Valid candle geometry
        # -------------------------------------------------

        data = data[
            (data["high"] >= data["low"])
            & (data["high"] >= data["open"])
            & (data["high"] >= data["close"])
            & (data["low"] <= data["open"])
            & (data["low"] <= data["close"])
        ]

        # -------------------------------------------------
        # Chronological order
        # -------------------------------------------------

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
    ) -> StructureResult:

        return StructureResult(

            detected=False,

            trend_bias="neutral",

            structure_state="undefined",

            current_index=-1,

            current_timestamp=None,

            swing_highs=[],

            swing_lows=[],

            swings=[],

            events=[],

            bos_events=[],

            mss_events=[],

            choch_events=[],

            latest_swing_high=None,

            latest_swing_low=None,

            latest_event=None,

            latest_bos=None,

            latest_mss=None,

            latest_choch=None,

            error=error,
        )


# =========================================================
# CONVENIENCE FUNCTIONS
# =========================================================

def analyze_structure(
    df: pd.DataFrame,
    current_index: Optional[int] = None,
    **kwargs,
) -> StructureResult:

    engine = MarketStructureEngine(
        **kwargs
    )

    return engine.analyze(
        df,
        current_index=current_index,
    )


def detect_structure(
    df: pd.DataFrame,
    current_index: Optional[int] = None,
    **kwargs,
) -> StructureResult:

    return analyze_structure(
        df,
        current_index=current_index,
        **kwargs,
    )


# =========================================================
# COMPATIBILITY ALIASES
# =========================================================

MarketStructureEngineV2 = (
    MarketStructureEngine
)

MarketStructureEngineV21 = (
    MarketStructureEngine
)


__all__ = [
    "SwingPoint",
    "StructureEvent",
    "StructureResult",
    "MarketStructureEngine",
    "MarketStructureEngineV2",
    "MarketStructureEngineV21",
    "analyze_structure",
    "detect_structure",
]