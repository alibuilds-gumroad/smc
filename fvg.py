from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

DEFAULT_LOOKBACK = 200
DEFAULT_ATR_PERIOD = 14

DEFAULT_MIN_GAP_ATR = 0.05
DEFAULT_MIN_DISPLACEMENT_ATR = 0.50

DEFAULT_MAX_ACTIVE_BARS = 100

# Fill thresholds
TOUCH_THRESHOLD = 0.0
MITIGATION_THRESHOLD = 0.50
FULL_FILL_THRESHOLD = 1.0

# Whether an FVG can be invalidated after full fill.
DEFAULT_INVALIDATE_ON_FULL_FILL = True


# ============================================================
# HELPERS
# ============================================================

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
        if np.isfinite(x):
            return x
    except Exception:
        pass
    return default


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _timestamp(df: pd.DataFrame, index: int) -> Any:
    if "timestamp" not in df.columns:
        return None

    if index < 0 or index >= len(df):
        return None

    return df.iloc[index]["timestamp"]


def _normalize_direction(value: Any) -> str:
    value = str(value).lower().strip()

    if value in {"bullish", "buy", "long", "up"}:
        return "bullish"

    if value in {"bearish", "sell", "short", "down"}:
        return "bearish"

    return value


# ============================================================
# ATR
# ============================================================

def calculate_atr(
    df: pd.DataFrame,
    period: int = DEFAULT_ATR_PERIOD,
) -> pd.Series:
    """
    Causal ATR.

    ATR at candle i only uses candles <= i.
    """

    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")

    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.rolling(
        window=period,
        min_periods=1,
    ).mean()


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class FairValueGap:
    id: str

    direction: str

    # Geometry
    gap_low: float
    gap_high: float
    gap_size: float

    # Formation
    first_index: int
    second_index: int
    third_index: int

    formation_index: int
    formation_timestamp: Any

    # IMPORTANT:
    # FVG becomes known only after third candle closes.
    confirmation_index: int
    confirmation_timestamp: Any

    # Quality
    atr: float
    gap_atr_ratio: float

    displacement: float = 0.0
    displacement_atr_ratio: float = 0.0

    score: float = 0.0

    # Lifecycle
    status: str = "active"

    touched: bool = False
    touch_index: Optional[int] = None
    touch_timestamp: Any = None

    mitigated: bool = False
    mitigation_index: Optional[int] = None
    mitigation_timestamp: Any = None

    filled: bool = False
    fill_index: Optional[int] = None
    fill_timestamp: Any = None

    invalidated: bool = False
    invalidation_index: Optional[int] = None
    invalidation_timestamp: Any = None

    # Latest observed fill state
    fill_ratio: float = 0.0
    max_fill_ratio: float = 0.0

    # Lifecycle age
    active_bars: int = 0

    # Context
    structure_type: Optional[str] = None
    structure_direction: Optional[str] = None

    @property
    def index(self) -> int:
        return self.formation_index

    @property
    def timestamp(self) -> Any:
        return self.formation_timestamp

    @property
    def confirmed(self) -> bool:
        return self.confirmation_index >= 0

    @property
    def active(self) -> bool:
        return (
            not self.filled
            and not self.invalidated
            and self.status in {
                "active",
                "touched",
                "mitigated",
            }
        )


@dataclass
class FVGResult:
    detected: bool

    latest: Optional[FairValueGap]

    best: Optional[FairValueGap]

    all_gaps: List[FairValueGap] = field(default_factory=list)

    bullish_gaps: List[FairValueGap] = field(default_factory=list)

    bearish_gaps: List[FairValueGap] = field(default_factory=list)

    active_gaps: List[FairValueGap] = field(default_factory=list)

    touched_gaps: List[FairValueGap] = field(default_factory=list)

    mitigated_gaps: List[FairValueGap] = field(default_factory=list)

    filled_gaps: List[FairValueGap] = field(default_factory=list)

    invalidated_gaps: List[FairValueGap] = field(default_factory=list)

    current_index: int = -1
    current_timestamp: Any = None

    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# FVG ENGINE
# ============================================================

class FVGEngine:

    def __init__(
        self,
        lookback: int = DEFAULT_LOOKBACK,
        atr_period: int = DEFAULT_ATR_PERIOD,
        min_gap_atr: float = DEFAULT_MIN_GAP_ATR,
        min_displacement_atr: float = DEFAULT_MIN_DISPLACEMENT_ATR,
        max_active_bars: int = DEFAULT_MAX_ACTIVE_BARS,
        invalidate_on_full_fill: bool = DEFAULT_INVALIDATE_ON_FULL_FILL,
    ):
        self.lookback = max(10, int(lookback))
        self.atr_period = max(1, int(atr_period))

        self.min_gap_atr = max(0.0, float(min_gap_atr))
        self.min_displacement_atr = max(
            0.0,
            float(min_displacement_atr),
        )

        self.max_active_bars = max(
            1,
            int(max_active_bars),
        )

        self.invalidate_on_full_fill = bool(
            invalidate_on_full_fill
        )

    # ========================================================
    # PUBLIC
    # ========================================================

    def analyze(
        self,
        df: pd.DataFrame,
        current_index: Optional[int] = None,
    ) -> FVGResult:

        try:
            data = self._validate_dataframe(df)

            if data.empty:
                return FVGResult(
                    detected=False,
                    latest=None,
                    best=None,
                    error="Empty dataframe",
                )

            if current_index is None:
                current_index = len(data) - 1

            current_index = max(
                0,
                min(int(current_index), len(data) - 1),
            )

            # =================================================
            # HARD CAUSAL SNAPSHOT
            # =================================================

            snapshot = data.iloc[
                : current_index + 1
            ].copy()

            atr = calculate_atr(
                snapshot,
                self.atr_period,
            )

            # =================================================
            # DETECT ONLY FVGs THAT ARE CONFIRMED NOW
            # =================================================

            gaps = self._detect_fvgs(
                snapshot,
                atr,
                current_index,
            )

            # =================================================
            # UPDATE LIFECYCLE ONLY WITH KNOWN CANDLES
            # =================================================

            for gap in gaps:
                self._update_lifecycle(
                    gap,
                    snapshot,
                    current_index,
                )

            # =================================================
            # SORT
            # =================================================

            gaps.sort(
                key=lambda x: (
                    x.confirmation_index,
                    x.score,
                )
            )

            bullish = [
                x for x in gaps
                if x.direction == "bullish"
            ]

            bearish = [
                x for x in gaps
                if x.direction == "bearish"
            ]

            active = [
                x for x in gaps
                if x.active
            ]

            touched = [
                x for x in gaps
                if x.touched
            ]

            mitigated = [
                x for x in gaps
                if x.mitigated
            ]

            filled = [
                x for x in gaps
                if x.filled
            ]

            invalidated = [
                x for x in gaps
                if x.invalidated
            ]

            latest = (
                gaps[-1]
                if gaps
                else None
            )

            best = (
                max(
                    active,
                    key=lambda x: x.score,
                )
                if active
                else (
                    max(
                        gaps,
                        key=lambda x: x.score,
                    )
                    if gaps
                    else None
                )
            )

            return FVGResult(
                detected=bool(gaps),
                latest=latest,
                best=best,
                all_gaps=gaps,
                bullish_gaps=bullish,
                bearish_gaps=bearish,
                active_gaps=active,
                touched_gaps=touched,
                mitigated_gaps=mitigated,
                filled_gaps=filled,
                invalidated_gaps=invalidated,
                current_index=current_index,
                current_timestamp=_timestamp(
                    snapshot,
                    current_index,
                ),
            )

        except Exception as exc:

            return FVGResult(
                detected=False,
                latest=None,
                best=None,
                current_index=(
                    -1
                    if current_index is None
                    else int(current_index)
                ),
                error=str(exc),
            )

    # ========================================================
    # FVG DETECTION
    # ========================================================

    def _detect_fvgs(
        self,
        df: pd.DataFrame,
        atr: pd.Series,
        current_index: int,
    ) -> List[FairValueGap]:

        n = len(df)

        if n < 3:
            return []

        start = max(
            2,
            n - self.lookback,
        )

        gaps: List[FairValueGap] = []

        for i in range(start, n):

            # -----------------------------------------------
            # 3-candle pattern
            #
            # candle A = i-2
            # candle B = i-1
            # candle C = i
            #
            # i is the FIRST moment the FVG is known.
            # -----------------------------------------------

            a = i - 2
            b = i - 1
            c = i

            a_high = _safe_float(df.iloc[a]["high"])
            a_low = _safe_float(df.iloc[a]["low"])

            b_open = _safe_float(df.iloc[b]["open"])
            b_close = _safe_float(df.iloc[b]["close"])

            c_high = _safe_float(df.iloc[c]["high"])
            c_low = _safe_float(df.iloc[c]["low"])

            c_close = _safe_float(df.iloc[c]["close"])

            atr_value = _safe_float(
                atr.iloc[c],
                0.0,
            )

            if atr_value <= 0:
                continue

            # =================================================
            # BULLISH FVG
            #
            # A.high < C.low
            #
            # Gap:
            # lower = A.high
            # upper = C.low
            # =================================================

            if a_high < c_low:

                gap_low = a_high
                gap_high = c_low
                gap_size = gap_high - gap_low

                if gap_size <= 0:
                    continue

                gap_atr_ratio = (
                    gap_size / atr_value
                )

                if gap_atr_ratio < self.min_gap_atr:
                    continue

                displacement = abs(
                    c_close - b_open
                )

                displacement_atr_ratio = (
                    displacement / atr_value
                )

                score = self._score(
                    gap_atr_ratio,
                    displacement_atr_ratio,
                )

                gaps.append(
                    FairValueGap(
                        id=f"fvg_bullish_{i}",
                        direction="bullish",
                        gap_low=gap_low,
                        gap_high=gap_high,
                        gap_size=gap_size,
                        first_index=a,
                        second_index=b,
                        third_index=c,
                        formation_index=c,
                        formation_timestamp=_timestamp(
                            df,
                            c,
                        ),
                        confirmation_index=c,
                        confirmation_timestamp=_timestamp(
                            df,
                            c,
                        ),
                        atr=atr_value,
                        gap_atr_ratio=gap_atr_ratio,
                        displacement=displacement,
                        displacement_atr_ratio=displacement_atr_ratio,
                        score=score,
                    )
                )

            # =================================================
            # BEARISH FVG
            #
            # A.low > C.high
            #
            # Gap:
            # lower = C.high
            # upper = A.low
            # =================================================

            elif a_low > c_high:

                gap_low = c_high
                gap_high = a_low
                gap_size = gap_high - gap_low

                if gap_size <= 0:
                    continue

                gap_atr_ratio = (
                    gap_size / atr_value
                )

                if gap_atr_ratio < self.min_gap_atr:
                    continue

                displacement = abs(
                    c_close - b_open
                )

                displacement_atr_ratio = (
                    displacement / atr_value
                )

                score = self._score(
                    gap_atr_ratio,
                    displacement_atr_ratio,
                )

                gaps.append(
                    FairValueGap(
                        id=f"fvg_bearish_{i}",
                        direction="bearish",
                        gap_low=gap_low,
                        gap_high=gap_high,
                        gap_size=gap_size,
                        first_index=a,
                        second_index=b,
                        third_index=c,
                        formation_index=c,
                        formation_timestamp=_timestamp(
                            df,
                            c,
                        ),
                        confirmation_index=c,
                        confirmation_timestamp=_timestamp(
                            df,
                            c,
                        ),
                        atr=atr_value,
                        gap_atr_ratio=gap_atr_ratio,
                        displacement=displacement,
                        displacement_atr_ratio=displacement_atr_ratio,
                        score=score,
                    )
                )

        return gaps

    # ========================================================
    # LIFECYCLE
    # ========================================================

    def _update_lifecycle(
        self,
        gap: FairValueGap,
        df: pd.DataFrame,
        current_index: int,
    ) -> None:

        confirmation = gap.confirmation_index

        # ----------------------------------------------------
        # NEVER inspect candles before confirmation.
        # ----------------------------------------------------

        if confirmation < 0:
            return

        if current_index <= confirmation:
            return

        last_index = min(
            current_index,
            confirmation + self.max_active_bars,
        )

        for i in range(
            confirmation + 1,
            last_index + 1,
        ):

            # Once terminal, stop processing.
            if gap.filled or gap.invalidated:
                break

            high = _safe_float(
                df.iloc[i]["high"]
            )

            low = _safe_float(
                df.iloc[i]["low"]
            )

            close = _safe_float(
                df.iloc[i]["close"]
            )

            # -----------------------------------------------
            # Calculate penetration into the FVG.
            # -----------------------------------------------

            fill_ratio = self._calculate_fill_ratio(
                gap,
                high,
                low,
            )

            fill_ratio = max(
                0.0,
                min(
                    1.0,
                    fill_ratio,
                ),
            )

            gap.fill_ratio = fill_ratio

            gap.max_fill_ratio = max(
                gap.max_fill_ratio,
                fill_ratio,
            )

            # -----------------------------------------------
            # ACTIVE AGE
            # -----------------------------------------------

            gap.active_bars = (
                i - confirmation
            )

            # -----------------------------------------------
            # INVALIDATION
            #
            # Bullish FVG:
            # price closes below gap_low
            #
            # Bearish FVG:
            # price closes above gap_high
            # -----------------------------------------------

            if self._is_invalidated(
                gap,
                close,
            ):

                gap.invalidated = True
                gap.invalidation_index = i
                gap.invalidation_timestamp = _timestamp(
                    df,
                    i,
                )
                gap.status = "invalidated"

                continue

            # -----------------------------------------------
            # TOUCH
            # -----------------------------------------------

            if (
                not gap.touched
                and fill_ratio > TOUCH_THRESHOLD
            ):

                gap.touched = True
                gap.touch_index = i
                gap.touch_timestamp = _timestamp(
                    df,
                    i,
                )

                gap.status = "touched"

            # -----------------------------------------------
            # MITIGATION
            # -----------------------------------------------

            if (
                not gap.mitigated
                and fill_ratio >= MITIGATION_THRESHOLD
            ):

                gap.mitigated = True
                gap.mitigation_index = i
                gap.mitigation_timestamp = _timestamp(
                    df,
                    i,
                )

                gap.status = "mitigated"

            # -----------------------------------------------
            # FULL FILL
            # -----------------------------------------------

            if fill_ratio >= FULL_FILL_THRESHOLD:

                gap.filled = True
                gap.fill_index = i
                gap.fill_timestamp = _timestamp(
                    df,
                    i,
                )

                gap.fill_ratio = 1.0

                if self.invalidate_on_full_fill:
                    gap.invalidated = True
                    gap.invalidation_index = i
                    gap.invalidation_timestamp = _timestamp(
                        df,
                        i,
                    )
                    gap.status = "filled"
                else:
                    gap.status = "filled"

                continue

            # -----------------------------------------------
            # ACTIVE
            # -----------------------------------------------

            if not gap.touched:
                gap.status = "active"

            elif gap.mitigated:
                gap.status = "mitigated"

            else:
                gap.status = "touched"

        # ----------------------------------------------------
        # Expiration
        # ----------------------------------------------------

        if (
            not gap.filled
            and not gap.invalidated
            and gap.active_bars >= self.max_active_bars
        ):
            gap.invalidated = True
            gap.invalidation_index = (
                confirmation
                + self.max_active_bars
            )

            if gap.invalidation_index <= current_index:
                gap.invalidation_timestamp = _timestamp(
                    df,
                    gap.invalidation_index,
                )

            gap.status = "expired"

    # ========================================================
    # FILL CALCULATION
    # ========================================================

    def _calculate_fill_ratio(
        self,
        gap: FairValueGap,
        high: float,
        low: float,
    ) -> float:

        gap_low = gap.gap_low
        gap_high = gap.gap_high
        gap_size = gap.gap_size

        if gap_size <= 0:
            return 0.0

        # ====================================================
        # BULLISH FVG
        #
        # Gap:
        #
        #       gap_high
        #       ─────────
        #          ↑
        #       untouched
        #
        #       gap_low
        #       ─────────
        #
        # Price enters from ABOVE.
        #
        # Penetration:
        #
        # high/low interaction measured from gap_high
        # downward toward gap_low.
        # ====================================================

        if gap.direction == "bullish":

            if low >= gap_high:
                return 0.0

            if low <= gap_low:
                return 1.0

            return (
                gap_high - low
            ) / gap_size

        # ====================================================
        # BEARISH FVG
        #
        # Price enters from BELOW.
        #
        # Penetration measured from gap_low upward toward
        # gap_high.
        # ====================================================

        if gap.direction == "bearish":

            if high <= gap_low:
                return 0.0

            if high >= gap_high:
                return 1.0

            return (
                high - gap_low
            ) / gap_size

        return 0.0

    # ========================================================
    # INVALIDATION
    # ========================================================

    def _is_invalidated(
        self,
        gap: FairValueGap,
        close: float,
    ) -> bool:

        if gap.direction == "bullish":
            return close < gap.gap_low

        if gap.direction == "bearish":
            return close > gap.gap_high

        return False

    # ========================================================
    # SCORE
    # ========================================================

    def _score(
        self,
        gap_atr_ratio: float,
        displacement_atr_ratio: float,
    ) -> float:

        gap_score = min(
            gap_atr_ratio * 4.0,
            4.0,
        )

        displacement_score = min(
            displacement_atr_ratio * 2.0,
            4.0,
        )

        return min(
            10.0,
            gap_score + displacement_score,
        )

    # ========================================================
    # VALIDATION
    # ========================================================

    def _validate_dataframe(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                "df must be a pandas DataFrame"
            )

        required = {
            "open",
            "high",
            "low",
            "close",
        }

        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                f"Missing columns: {sorted(missing)}"
            )

        data = df.copy()

        for col in [
            "open",
            "high",
            "low",
            "close",
        ]:
            data[col] = pd.to_numeric(
                data[col],
                errors="coerce",
            )

        data = data.dropna(
            subset=[
                "open",
                "high",
                "low",
                "close",
            ]
        )

        if "timestamp" in data.columns:
            data["timestamp"] = pd.to_datetime(
                data["timestamp"],
                errors="coerce",
            )

            data = data.sort_values(
                "timestamp",
                kind="stable",
            )

        data = data.reset_index(
            drop=True
        )

        return data


# ============================================================
# CONVENIENCE API
# ============================================================

def detect_fvg(
    df: pd.DataFrame,
    current_index: Optional[int] = None,
    lookback: int = DEFAULT_LOOKBACK,
    atr_period: int = DEFAULT_ATR_PERIOD,
    min_gap_atr: float = DEFAULT_MIN_GAP_ATR,
    min_displacement_atr: float = DEFAULT_MIN_DISPLACEMENT_ATR,
) -> FVGResult:

    engine = FVGEngine(
        lookback=lookback,
        atr_period=atr_period,
        min_gap_atr=min_gap_atr,
        min_displacement_atr=min_displacement_atr,
    )

    return engine.analyze(
        df,
        current_index=current_index,
    )


# Backward-compatible alias
FairValueGapEngine = FVGEngine