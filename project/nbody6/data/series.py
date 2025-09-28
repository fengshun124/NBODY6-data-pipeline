import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple, Union

import pandas as pd

from nbody6.calc.summary import summarize_timestamp_stats
from nbody6.data.snapshot import Snapshot


@dataclass(slots=True)
class SnapshotSeries:
    snapshot_dict: Dict[float, Snapshot]

    # caches
    _cache_summary: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)
    _cache_binary_annular: Optional[pd.DataFrame] = field(
        default=None, init=False, repr=False
    )

    _parent_invalidator: Optional[Callable[[], None]] = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self):
        tolerance = 2e-2
        for timestamp, snapshot in self.snapshot_dict.items():
            if abs(timestamp - snapshot.time) > tolerance:
                raise ValueError(
                    f"Timestamp {timestamp} does not match snapshot time {snapshot.time}."
                )
        self._bind_children()

    # caching and invalidation
    def _clear_cache(self):
        self._cache_summary = None
        self._cache_binary_annular = None
        # bubble up
        if self._parent_invalidator is not None:
            self._parent_invalidator()

    def __setattr__(self, name: str, value) -> None:
        if hasattr(self, "_cache_summary") and name == "snapshot_dict":
            # clear before rebinding
            try:
                object.__setattr__(self, name, value)
            finally:
                self._bind_children()
                self._clear_cache()
            return
        object.__setattr__(self, name, value)

    def _bind_children(self):
        # Register invalidators so that child mutations invalidate this series
        for snapshot in self.snapshot_dict.values():
            try:
                snapshot._parent_invalidator = self._clear_cache
            except Exception:
                pass

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f"num_snapshots={len(self.snapshot_dict)}, "
            f"timestamp_stats={self.timestamp_stats}"
            ")"
        )

    def __len__(self) -> int:
        return len(self.snapshot_dict)

    def __iter__(self) -> Iterator[Tuple[float, Snapshot]]:
        for timestamp, snapshot in sorted(self.snapshot_dict.items()):
            yield (timestamp, snapshot)

    def __getitem__(self, timestamp: float) -> Snapshot:
        return self.snapshot_dict[timestamp]

    @property
    def timestamps(self) -> List[float]:
        return sorted(self.snapshot_dict.keys())

    @property
    def timestamp_stats(self) -> Dict[str, Dict[str, Optional[float]]]:
        return summarize_timestamp_stats(self.timestamps)

    def to_dict(self, is_materialize: bool = True) -> Dict:
        return {
            timestamp: snapshot.to_dict(is_materialize=is_materialize)
            for timestamp, snapshot in self
        }

    def to_pickle(
        self, filepath: Union[str, Path], enforce_overwrite: bool = False
    ) -> None:
        filepath = Path(filepath).resolve()
        if filepath.exists() and not enforce_overwrite:
            raise FileExistsError(f"{filepath} already exists.")
        with open(filepath, "wb") as f:
            pickle.dump(self.to_dict(is_materialize=True), f)

    @classmethod
    def from_dict(cls, data: Dict) -> "SnapshotSeries":
        obj = cls(
            snapshot_dict={
                float(timestamp): Snapshot.from_dict(snapshot)
                for timestamp, snapshot in data.items()
            }
        )
        return obj

    @classmethod
    def from_pickle(cls, filepath: Union[str, Path]) -> "SnapshotSeries":
        filepath = Path(filepath)
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        return cls.from_dict(data)

    # overall summary
    @property
    def summary(self) -> pd.DataFrame:
        if self._cache_summary is None:
            self._cache_summary = self._summarize()
        return self._cache_summary

    def _summarize(self) -> pd.DataFrame:
        def _summarize_snapshot():
            for t, s in sorted(self.snapshot_dict.items()):
                snapshot_summary = s.summary.copy()
                snapshot_summary.insert(0, "timestamp", t)
                yield snapshot_summary

        if not self.snapshot_dict:
            return pd.DataFrame()

        return pd.concat(_summarize_snapshot(), ignore_index=True)

    # binary annular statistics
    @property
    def binary_annular_statistics(self) -> pd.DataFrame:
        if self._cache_binary_annular is None:
            self._cache_binary_annular = self._compute_binary_annular_statistics()
        return self._cache_binary_annular

    def _compute_binary_annular_statistics(self) -> pd.DataFrame:
        if not self.snapshot_dict:
            return pd.DataFrame()

        results = []
        for t, snapshot in self:
            stats = snapshot.binary_annular_statistics
            if stats is None or stats.empty:
                continue
            df = stats.copy()
            df.insert(0, "timestamp", t)
            results.append(df)
        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()
