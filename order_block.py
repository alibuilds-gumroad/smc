"""
smc/order_block.py
============================================================
SMC Order Block Engine - V2.1
============================================================

Responsibilities
----------------
1. Detect bullish / bearish Order Blocks
2. Use confirmed BOS / MSS / CHOCH as structural context
3. Detect the last opposing candle before displacement
4. Enforce strict confirmation_index
5. Prevent future-data leakage
6. Track mitigation / invalidation
7. Provide deterministic output
8. Candle-by-candle compatible
9. Compatible with zones.py V2.1

Definition used here
--------------------
Bullish OB:
    Last bearish candle before a confirmed bullish
    displacement / structure break.

Bearish OB:
    Last bullish candle before a confirmed bearish
    displacement / structure break.

Important
---------
The OB is NOT considered known at the candle that created it.

Example:

    displacement candle = 120
    OB source candle    = 117

The OB becomes known at:

    confirmation_index = 120

Therefore it cannot be used for candles < 120.

This is intentionally conservative and backtest-safe.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import math

import pandas as pd


# ============================================================
# Defaults
# ============================================================

DEFAULT_LOOKBACK = 150
DEFAULT_ATR_PERIOD = 14

DEFAULT_MIN_BODY_ATR = 0.10
DEFAULT_DISPLACEMENT_ATR = 0.50

DEFAULT_MAX_OB_ATR = 2.0
DEFAULT_INVALIDATION_ATR = 0.10

DEFAULT_MIN_SCORE = 20.0


# ============================================================
# Dataclasses
# ============================================================

@dataclass
class OrderBlock:
    """
    Confirmed Order Block.

    direction:
        bullish / bearish

    Source candle:
        The opposing candle immediately before displacement.

    confirmation_index:
        Candle at which the displacement / structure event
        confirms the OB.
    """

    id: str

    direction: str

    index: int
    timestamp: Any

    confirmation_index: int
    confirmation_timestamp: Any

    low: float
    high: float

    open: float
    close: float

    body: float
    range: float

    body_atr_ratio: float
    range_atr_ratio: float

    atr: float

    score: float

    structure_type: Optional[str] = None
    structure_direction: Optional[str] = None

    structure_event_index: Optional[int] = None
    structure_event_timestamp: Any = None

    displacement_atr_ratio: Optional[float] = None

    mitigated: bool = False
    filled: bool = False
    invalidated: bool = False

    mitigation_index: Optional[int] = None
    mitigation_timestamp: Any = None

    fill_index: Optional[int] = None
    fill_timestamp: Any = None

    invalidation_index: Optional[int] = None
    invalidation_timestamp: Any = None

    def is_available(self, current_index: int) -> bool:
        return self.confirmation_index <= current_index

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OrderBlockResult:
    detected: bool = False

    latest: Optional[OrderBlock] = None
    best: Optional[OrderBlock] = None

    order_blocks: List[OrderBlock] = field(
        default_factory=list
    )

    bullish_blocks: List[OrderBlock] = field(
        default_factory=list
    )

    bearish_blocks: List[OrderBlock] = field(
        default_factory=list
    )

    active_blocks: List[OrderBlock] = field(
        default_factory=list
    )

    mitigated_blocks: List[OrderBlock] = field(
        default_factory=list
    )

    invalidated_blocks: List[OrderBlock] = field(
        default_factory=list
    )

    current_index: Optional[int] = None
    current_timestamp: Any = None

    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def all_order_blocks(self) -> List[OrderBlock]:
        """
        Alias used by downstream modules.
        """
        return self.order_blocks

    def confirmed_blocks(
        self,
        current_index: Optional[int] = None,
    ) -> List[OrderBlock]:

        if current_index is None:
            current_index = self.current_index

        if current_index is None:
            return list(self.order_blocks)

        return [
            block
            for block in self.order_blocks
            if block.confirmation_index <= current_index
        ]


# ============================================================
# Engine
# ============================================================

class OrderBlockEngine:

    def __init__(
        self,
        lookback: int = DEFAULT_LOOKBACK,
        atr_period: int = DEFAULT_ATR_PERIOD,
        min_body_atr: float = DEFAULT_MIN_BODY_ATR,
        displacement_atr: float = DEFAULT_DISPLACEMENT_ATR,
        max_ob_atr: float = DEFAULT_MAX_OB_ATR,
        invalidation_atr: float = DEFAULT_INVALIDATION_ATR,
        min_score: float = DEFAULT_MIN_SCORE,
    ):

        if lookback < 20:
            raise ValueError(
                "lookback should be >= 20"
            )

        if atr_period < 1:
            raise ValueError(
                "atr_period must be >= 1"
            )

        self.lookback = int(lookback)
        self.atr_period = int(atr_period)

        self.min_body_atr = float(
            min_body_atr
        )

        self.displacement_atr = float(
            displacement_atr
        )

        self.max_ob_atr = float(
            max_ob_atr
        )

        self.invalidation_atr = float(
            invalidation_atr
        )

        self.min_score = float(
            min_score
        )

    # ========================================================
    # Public API
    # ========================================================

    def analyze(
        self,
        df: pd.DataFrame,
        current_index: Optional[int] = None,
        structure_result=None,
        liquidity_result=None,
    ) -> OrderBlockResult:

        try:
            data = self._validate_dataframe(df)

            if data.empty:
                return OrderBlockResult(
                    detected=False,
                    error="Empty dataframe",
                )

            if current_index is None:
                current_index = len(data) - 1

            current_index = min(
                max(0, int(current_index)),
                len(data) - 1,
            )

            work = data.iloc[
                : current_index + 1
            ].copy()

            if len(work) < 10:
                return OrderBlockResult(
                    detected=False,
                    current_index=current_index,
                    current_timestamp=self._timestamp(
                        data,
                        current_index,
                    ),
                    error="Not enough candles",
                )

            atr = self._calculate_atr(work)

            # ------------------------------------------------
            # Structure is the primary source of OB confirmation.
            # ------------------------------------------------

            if structure_result is None:

                from smc.structure import (
                    MarketStructureEngine,
                )

                structure_engine = (
                    MarketStructureEngine(
                        lookback=self.lookback,
                        atr_period=self.atr_period,
                    )
                )

                structure_result = (
                    structure_engine.analyze(
                        work,
                        current_index=current_index,
                    )
                )

            events = getattr(
                structure_result,
                "events",
                [],
            )

            events = [
                event
                for event in events
                if self._event_available(
                    event,
                    current_index,
                )
            ]

            # ------------------------------------------------
            # Detect OBs from confirmed structural events.
            # ------------------------------------------------

            blocks = self._detect_from_structure(
                work,
                atr,
                events,
                current_index,
            )

            # ------------------------------------------------
            # Optional liquidity context.
            # ------------------------------------------------

            blocks = self._apply_liquidity_context(
                blocks,
                liquidity_result,
            )

            # ------------------------------------------------
            # Lifecycle.
            # ------------------------------------------------

            blocks = self._update_lifecycle(
                work,
                blocks,
                atr,
                current_index,
            )

            blocks = [
                block
                for block in blocks
                if block.score >= self.min_score
            ]

            blocks.sort(
                key=lambda block: (
                    block.confirmation_index,
                    block.index,
                )
            )

            bullish = [
                block
                for block in blocks
                if block.direction == "bullish"
            ]

            bearish = [
                block
                for block in blocks
                if block.direction == "bearish"
            ]

            active = [
                block
                for block in blocks
                if (
                    not block.invalidated
                    and not block.filled
                )
            ]

            mitigated = [
                block
                for block in blocks
                if block.mitigated
            ]

            invalidated = [
                block
                for block in blocks
                if block.invalidated
            ]

            latest = (
                blocks[-1]
                if blocks
                else None
            )

            best = self._select_best(
                active,
                current_index,
            )

            return OrderBlockResult(
                detected=bool(blocks),
                latest=latest,
                best=best,
                order_blocks=blocks,
                bullish_blocks=bullish,
                bearish_blocks=bearish,
                active_blocks=active,
                mitigated_blocks=mitigated,
                invalidated_blocks=invalidated,
                current_index=current_index,
                current_timestamp=self._timestamp(
                    data,
                    current_index,
                ),
            )

        except Exception as exc:

            return OrderBlockResult(
                detected=False,
                current_index=current_index,
                error=f"{type(exc).__name__}: {exc}",
            )

    # ========================================================
    # Detect OB from Structure
    # ========================================================

    def _detect_from_structure(
        self,
        df: pd.DataFrame,
        atr: pd.Series,
        events: List[Any],
        current_index: int,
    ) -> List[OrderBlock]:

        blocks: List[OrderBlock] = []

        min_index = max(
            1,
            current_index - self.lookback + 1,
        )

        for event in events:

            event_index = getattr(
                event,
                "index",
                None,
            )

            if event_index is None:
                continue

            event_index = int(
                event_index
            )

            if event_index <= min_index:
                continue

            if event_index > current_index:
                continue

            direction = getattr(
                event,
                "direction",
                None,
            )

            if direction not in {
                "bullish",
                "bearish",
            }:
                continue

            # ------------------------------------------------
            # Only use confirmed structural events.
            # ------------------------------------------------

            confirmation_index = getattr(
                event,
                "confirmation_index",
                None,
            )

            if confirmation_index is None:
                continue

            confirmation_index = int(
                confirmation_index
            )

            if confirmation_index > current_index:
                continue

            # ------------------------------------------------
            # Find last opposing candle before the
            # displacement / structure break.
            # ------------------------------------------------

            source_index = (
                self._find_source_candle(
                    df,
                    event_index,
                    direction,
                )
            )

            if source_index is None:
                continue

            if source_index < min_index:
                continue

            candle = df.iloc[
                source_index
            ]

            open_price = float(
                candle["open"]
            )

            close_price = float(
                candle["close"]
            )

            high = float(
                candle["high"]
            )

            low = float(
                candle["low"]
            )

            body = abs(
                close_price - open_price
            )

            candle_range = (
                high - low
            )

            atr_value = self._safe_atr(
                atr,
                source_index,
            )

            if atr_value <= 0:
                continue

            body_atr_ratio = (
                body / atr_value
            )

            range_atr_ratio = (
                candle_range / atr_value
            )

            # ------------------------------------------------
            # Reject abnormally large candles.
            # ------------------------------------------------

            if (
                range_atr_ratio
                > self.max_ob_atr
            ):
                continue

            # ------------------------------------------------
            # Weak source candles are allowed, but extremely
            # tiny bodies are less useful.
            # ------------------------------------------------

            if (
                body_atr_ratio
                < self.min_body_atr
            ):
                # Do not hard reject if it is a valid
                # opposing candle. Instead reduce score.
                body_quality = 0.0
            else:
                body_quality = min(
                    10.0,
                    body_atr_ratio * 10.0,
                )

            displacement_atr_ratio = getattr(
                event,
                "displacement_atr",
                None,
            )

            if (
                displacement_atr_ratio
                is None
            ):
                displacement_atr_ratio = (
                    self._calculate_event_displacement(
                        df,
                        event,
                        atr,
                    )
                )

            # ------------------------------------------------
            # If displacement is explicitly measurable and
            # too weak, don't create a high-quality OB.
            # ------------------------------------------------

            if (
                displacement_atr_ratio is not None
                and displacement_atr_ratio
                < self.displacement_atr
            ):
                displacement_quality = 0.0
            else:
                displacement_quality = min(
                    20.0,
                    (
                        float(
                            displacement_atr_ratio
                            or 0.0
                        )
                        * 10.0
                    ),
                )

            structure_quality = {
                "BOS": 15.0,
                "MSS": 20.0,
                "CHOCH": 25.0,
            }.get(
                getattr(event, "type", None),
                10.0,
            )

            score = min(
                100.0,
                30.0
                + body_quality
                + displacement_quality
                + structure_quality,
            )

            block_id = (
                f"ob_{direction}_"
                f"{source_index}_"
                f"{event_index}"
            )

            source_timestamp = (
                self._timestamp(
                    df,
                    source_index,
                )
            )

            confirmation_timestamp = (
                self._timestamp(
                    df,
                    confirmation_index,
                )
            )

            blocks.append(
                OrderBlock(
                    id=block_id,
                    direction=direction,

                    index=source_index,
                    timestamp=source_timestamp,

                    confirmation_index=(
                        confirmation_index
                    ),
                    confirmation_timestamp=(
                        confirmation_timestamp
                    ),

                    low=low,
                    high=high,

                    open=open_price,
                    close=close_price,

                    body=body,
                    range=candle_range,

                    body_atr_ratio=(
                        body_atr_ratio
                    ),
                    range_atr_ratio=(
                        range_atr_ratio
                    ),

                    atr=atr_value,

                    score=score,

                    structure_type=getattr(
                        event,
                        "type",
                        None,
                    ),

                    structure_direction=(
                        direction
                    ),

                    structure_event_index=(
                        event_index
                    ),

                    structure_event_timestamp=(
                        getattr(
                            event,
                            "timestamp",
                            None,
                        )
                    ),

                    displacement_atr_ratio=(
                        displacement_atr_ratio
                    ),
                )
            )

        return self._deduplicate(
            blocks
        )

    # ========================================================
    # Source Candle
    # ========================================================

    @staticmethod
    def _find_source_candle(
        df: pd.DataFrame,
        event_index: int,
        direction: str,
    ) -> Optional[int]:

        """
        Bullish displacement:
            find last bearish candle.

        Bearish displacement:
            find last bullish candle.

        Search backward up to 5 candles.
        """

        start = max(
            0,
            event_index - 5,
        )

        for i in range(
            event_index - 1,
            start - 1,
            -1,
        ):

            candle = df.iloc[i]

            open_price = float(
                candle["open"]
            )

            close_price = float(
                candle["close"]
            )

            if direction == "bullish":
                # Last bearish candle.
                if close_price < open_price:
                    return i

            elif direction == "bearish":
                # Last bullish candle.
                if close_price > open_price:
                    return i

        return None

    # ========================================================
    # Displacement
    # ========================================================

    def _calculate_event_displacement(
        self,
        df: pd.DataFrame,
        event: Any,
        atr: pd.Series,
    ) -> Optional[float]:

        event_index = getattr(
            event,
            "index",
            None,
        )

        if event_index is None:
            return None

        event_index = int(
            event_index
        )

        if event_index < 0:
            return None

        if event_index >= len(df):
            return None

        candle = df.iloc[
            event_index
        ]

        open_price = float(
            candle["open"]
        )

        close_price = float(
            candle["close"]
        )

        atr_value = self._safe_atr(
            atr,
            event_index,
        )

        if atr_value <= 0:
            return None

        direction = getattr(
            event,
            "direction",
            None,
        )

        if direction == "bullish":
            displacement = (
                close_price - open_price
            )

        elif direction == "bearish":
            displacement = (
                open_price - close_price
            )

        else:
            return None

        return (
            displacement / atr_value
        )

    # ========================================================
    # Liquidity Context
    # ========================================================

    def _apply_liquidity_context(
        self,
        blocks: List[OrderBlock],
        liquidity_result,
    ) -> List[OrderBlock]:

        if not blocks:
            return blocks

        if liquidity_result is None:
            return blocks

        sweeps = getattr(
            liquidity_result,
            "sweeps",
            [],
        )

        for block in blocks:

            candidates = []

            for sweep in sweeps:

                sweep_confirmation = getattr(
                    sweep,
                    "confirmation_index",
                    None,
                )

                if sweep_confirmation is None:
                    continue

                if (
                    int(sweep_confirmation)
                    > int(block.confirmation_index)
                ):
                    continue

                reaction = getattr(
                    sweep,
                    "reaction",
                    None,
                )

                if reaction != block.direction:
                    continue

                candidates.append(
                    sweep
                )

            if not candidates:
                continue

            candidates.sort(
                key=lambda x: (
                    getattr(
                        x,
                        "confirmation_index",
                        -1,
                    ),
                    getattr(
                        x,
                        "sweep_index",
                        -1,
                    ),
                )
            )

            latest = candidates[-1]

            # Liquidity confluence bonus.
            block.score = min(
                100.0,
                block.score + 10.0,
            )

        return blocks

    # ========================================================
    # Lifecycle
    # ========================================================

    def _update_lifecycle(
        self,
        df: pd.DataFrame,
        blocks: List[OrderBlock],
        atr: pd.Series,
        current_index: int,
    ) -> List[OrderBlock]:

        for block in blocks:

            start = max(
                block.confirmation_index + 1,
                block.index + 1,
            )

            end = min(
                current_index,
                len(df) - 1,
            )

            if start > end:
                continue

            invalidation_buffer = (
                block.atr
                * self.invalidation_atr
            )

            for i in range(
                start,
                end + 1,
            ):

                candle = df.iloc[i]

                high = float(
                    candle["high"]
                )

                low = float(
                    candle["low"]
                )

                close = float(
                    candle["close"]
                )

                # ------------------------------------------------
                # Bullish OB
                # ------------------------------------------------

                if block.direction == "bullish":

                    invalidation_price = (
                        block.low
                        - invalidation_buffer
                    )

                    if close < invalidation_price:

                        block.invalidated = True
                        block.invalidation_index = i
                        block.invalidation_timestamp = (
                            self._timestamp(
                                df,
                                i,
                            )
                        )

                        break

                    touched = (
                        low <= block.high
                        and high >= block.low
                    )

                    if touched:
                        block.mitigated = True

                        if (
                            block.mitigation_index
                            is None
                        ):
                            block.mitigation_index = i
                            block.mitigation_timestamp = (
                                self._timestamp(
                                    df,
                                    i,
                                )
                            )

                    # Full fill.
                    if low <= block.low:
                        block.filled = True

                        if (
                            block.fill_index
                            is None
                        ):
                            block.fill_index = i
                            block.fill_timestamp = (
                                self._timestamp(
                                    df,
                                    i,
                                )
                            )

                # ------------------------------------------------
                # Bearish OB
                # ------------------------------------------------

                elif block.direction == "bearish":

                    invalidation_price = (
                        block.high
                        + invalidation_buffer
                    )

                    if close > invalidation_price:

                        block.invalidated = True
                        block.invalidation_index = i
                        block.invalidation_timestamp = (
                            self._timestamp(
                                df,
                                i,
                            )
                        )

                        break

                    touched = (
                        low <= block.high
                        and high >= block.low
                    )

                    if touched:
                        block.mitigated = True

                        if (
                            block.mitigation_index
                            is None
                        ):
                            block.mitigation_index = i
                            block.mitigation_timestamp = (
                                self._timestamp(
                                    df,
                                    i,
                                )
                            )

                    # Full fill.
                    if high >= block.high:
                        block.filled = True

                        if (
                            block.fill_index
                            is None
                        ):
                            block.fill_index = i
                            block.fill_timestamp = (
                                self._timestamp(
                                    df,
                                    i,
                                )
                            )

        return blocks

    # ========================================================
    # Best Block
    # ========================================================

    @staticmethod
    def _select_best(
        blocks: List[OrderBlock],
        current_index: int,
    ) -> Optional[OrderBlock]:

        if not blocks:
            return None

        available = [
            block
            for block in blocks
            if (
                block.confirmation_index
                <= current_index
            )
        ]

        if not available:
            return None

        # Score first, freshness second.
        available.sort(
            key=lambda block: (
                block.score,
                block.confirmation_index,
            ),
            reverse=True,
        )

        return available[0]

    # ========================================================
    # Deduplication
    # ========================================================

    @staticmethod
    def _deduplicate(
        blocks: List[OrderBlock],
    ) -> List[OrderBlock]:

        seen = set()
        result = []

        for block in blocks:

            key = block.id

            if key in seen:
                continue

            seen.add(key)
            result.append(block)

        return result

    # ========================================================
    # Event Validation
    # ========================================================

    @staticmethod
    def _event_available(
        event: Any,
        current_index: int,
    ) -> bool:

        confirmation_index = getattr(
            event,
            "confirmation_index",
            None,
        )

        if confirmation_index is None:
            return False

        try:
            return (
                int(confirmation_index)
                <= int(current_index)
            )

        except Exception:
            return False

    # ========================================================
    # ATR
    # ========================================================

    def _calculate_atr(
        self,
        df: pd.DataFrame,
    ) -> pd.Series:

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

        return tr.ewm(
            alpha=1 / self.atr_period,
            adjust=False,
            min_periods=1,
        ).mean()

    # ========================================================
    # Helpers
    # ========================================================

    @staticmethod
    def _safe_atr(
        atr: pd.Series,
        index: int,
    ) -> float:

        try:
            value = float(
                atr.iloc[index]
            )

            if not math.isfinite(value):
                return 0.0

            return value

        except Exception:
            return 0.0

    @staticmethod
    def _timestamp(
        df: pd.DataFrame,
        index: int,
    ) -> Any:

        if index < 0 or index >= len(df):
            return None

        if "timestamp" in df.columns:
            return df.iloc[index]["timestamp"]

        if "time" in df.columns:
            return df.iloc[index]["time"]

        if isinstance(
            df.index,
            pd.DatetimeIndex,
        ):
            return df.index[index]

        return index

    @staticmethod
    def _validate_dataframe(
        df: pd.DataFrame,
    ) -> pd.DataFrame:

        if not isinstance(
            df,
            pd.DataFrame,
        ):
            raise TypeError(
                "df must be a pandas DataFrame"
            )

        required = {
            "open",
            "high",
            "low",
            "close",
        }

        lower_columns = {
            str(column).lower()
            for column in df.columns
        }

        missing = (
            required - lower_columns
        )

        if missing:
            raise ValueError(
                "Missing OHLC columns: "
                f"{sorted(missing)}"
            )

        rename_map = {}

        for column in df.columns:

            lower = str(column).lower()

            if lower in {
                "open",
                "high",
                "low",
                "close",
                "volume",
                "tick_volume",
                "timestamp",
                "time",
            }:
                rename_map[column] = lower

        data = df.rename(
            columns=rename_map
        ).copy()

        for column in [
            "open",
            "high",
            "low",
            "close",
        ]:

            data[column] = pd.to_numeric(
                data[column],
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

            timestamps = pd.to_datetime(
                data["timestamp"],
                errors="coerce",
            )

            if timestamps.notna().all():

                data["timestamp"] = timestamps

                if not timestamps.is_monotonic_increasing:
                    data = data.sort_values(
                        "timestamp"
                    )

        elif "time" in data.columns:

            timestamps = pd.to_datetime(
                data["time"],
                errors="coerce",
            )

            if timestamps.notna().all():

                data["time"] = timestamps

                if not timestamps.is_monotonic_increasing:
                    data = data.sort_values(
                        "time"
                    )

        return data


# ============================================================
# Convenience API
# ============================================================

def detect_order_blocks(
    df: pd.DataFrame,
    current_index: Optional[int] = None,
    **kwargs,
) -> OrderBlockResult:

    engine = OrderBlockEngine(
        **kwargs
    )

    return engine.analyze(
        df,
        current_index=current_index,
    )


def analyze_order_blocks(
    df: pd.DataFrame,
    current_index: Optional[int] = None,
    **kwargs,
) -> OrderBlockResult:

    return detect_order_blocks(
        df=df,
        current_index=current_index,
        **kwargs,
    )


# Backward-compatible aliases
OrderBlockEngineV2 = OrderBlockEngine