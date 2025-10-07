import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple, Union

import joblib
import pandas as pd

from nbody6.calc.summary import summarize_timestamp_stats
from nbody6.data.snapshot import Snapshot


@dataclass(slots=True)
class SnapshotSeries:
    snapshot_dict: Dict[float, Snapshot]

    # caches
    _cache_stats: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)
    _cache_bin_annular_stats: Optional[pd.DataFrame] = field(
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
        self._cache_stats = None
        self._cache_bin_annular_stats = None
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

    def to_joblib(
        self, filepath: Union[str, Path], enforce_overwrite: bool = False
    ) -> None:
        filepath = Path(filepath).resolve()
        if filepath.exists() and not enforce_overwrite:
            raise FileExistsError(f"{filepath} already exists.")
        joblib.dump(self.to_dict(is_materialize=False), filepath, compress=3)

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

    @classmethod
    def from_joblib(cls, filepath: Union[str, Path]) -> "SnapshotSeries":
        filepath = Path(filepath)
        data = joblib.load(filepath)
        return cls.from_dict(data)

    # overall statistics
    @property
    def statistics(self) -> pd.DataFrame:
        if self._cache_stats is None:
            self._cache_stats = self._calc_stats()
        return self._cache_stats

    def _calc_stats(self) -> pd.DataFrame:
        if not self.snapshot_dict:
            return pd.DataFrame()

        stats_dfs = []
        for timestamp, snapshot in sorted(self.snapshot_dict.items()):
            snapshot_stats = snapshot.statistics.copy()
            snapshot_stats.insert(0, "timestamp", timestamp)
            stats_dfs.append(snapshot_stats)

        return pd.concat(stats_dfs, ignore_index=True)

    # annular statistics
    @property
    def annular_statistics(self) -> pd.DataFrame:
        if self._cache_bin_annular_stats is None:
            self._cache_bin_annular_stats = self._calc_annular_stats()
        return self._cache_bin_annular_stats

    def _calc_annular_stats(self) -> pd.DataFrame:
        if not self.snapshot_dict:
            return pd.DataFrame()

        annular_stats_dfs = []
        for timestamp, snapshot in sorted(self.snapshot_dict.items()):
            snapshot_annular = snapshot.annular_statistics
            if snapshot_annular.empty:
                continue
            snapshot_annular_copy = snapshot_annular.copy()
            snapshot_annular_copy.insert(0, "timestamp", timestamp)
            annular_stats_dfs.append(snapshot_annular_copy)

        return (
            pd.concat(annular_stats_dfs, ignore_index=True)
            if annular_stats_dfs
            else pd.DataFrame()
        )
