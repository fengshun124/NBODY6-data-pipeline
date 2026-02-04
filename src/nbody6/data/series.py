import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import joblib
import pandas as pd

from nbody6.calc.summary import summarize_timestamp_stats
from nbody6.data.snapshot import PseudoObservedSnapshot, Snapshot


@dataclass(slots=True)
class SnapshotSeries:
    snapshot_dict: dict[float, Snapshot]

    # caches
    _cache_stats: pd.DataFrame | None = field(default=None, init=False, repr=False)
    _cache_bin_annular_stats: pd.DataFrame | None = field(
        default=None, init=False, repr=False
    )

    _parent_invalidator: Callable[[], None] | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        tolerance = 2e-2
        for timestamp, snapshot in self.snapshot_dict.items():
            if abs(timestamp - snapshot.time) > tolerance:
                raise ValueError(
                    f"Timestamp {timestamp} does not match snapshot time {snapshot.time}."
                )
        self._bind_children()

    # caching and invalidation
    def _clear_cache(self) -> None:
        self._cache_stats = None
        self._cache_bin_annular_stats = None
        # bubble up
        if self._parent_invalidator is not None:
            self._parent_invalidator()

    def __setattr__(self, name: str, value) -> None:
        if hasattr(self, "_cache_stats") and name == "snapshot_dict":
            # clear before rebinding
            try:
                object.__setattr__(self, name, value)
            finally:
                self._bind_children()
                self._clear_cache()
            return
        object.__setattr__(self, name, value)

    def _bind_children(self) -> None:
        # Register invalidators so that child mutations invalidate this series
        for snapshot in self.snapshot_dict.values():
            try:
                snapshot._parent_invalidator = self._clear_cache
            except Exception:
                pass

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"num_snapshots={len(self.snapshot_dict)}, "
            f"timestamp_stats={self.timestamp_stats}"
            ")"
        )

    def __len__(self) -> int:
        return len(self.snapshot_dict)

    def __iter__(self) -> Iterator[tuple[float, Snapshot]]:
        for timestamp, snapshot in sorted(self.snapshot_dict.items()):
            yield (timestamp, snapshot)

    def __getitem__(self, timestamp: float) -> Snapshot:
        return self.snapshot_dict[timestamp]

    @property
    def timestamps(self) -> list[float]:
        return sorted(self.snapshot_dict.keys())

    @property
    def timestamp_stats(self) -> dict[str, dict[str, float | None]]:
        return summarize_timestamp_stats(self.timestamps)

    def to_dict(self, is_materialize: bool = True) -> dict:
        return {
            timestamp: snapshot.to_dict(is_materialize=is_materialize)
            for timestamp, snapshot in self
        }

    def to_pickle(self, filepath: Path | str, enforce_overwrite: bool = False) -> None:
        filepath = Path(filepath).resolve()
        if filepath.exists() and not enforce_overwrite:
            raise FileExistsError(f"{filepath} already exists.")

        tmp_filepath = filepath.with_suffix(filepath.suffix + ".tmp")
        try:
            with open(tmp_filepath, "wb") as f:
                pickle.dump(self.to_dict(is_materialize=True), f)
            tmp_filepath.replace(filepath)
        finally:
            tmp_filepath.unlink(missing_ok=True)

    def to_joblib(self, filepath: Path | str, enforce_overwrite: bool = False) -> None:
        filepath = Path(filepath).resolve()
        if filepath.exists() and not enforce_overwrite:
            raise FileExistsError(f"{filepath} already exists.")

        tmp_filepath = filepath.with_suffix(filepath.suffix + ".tmp")
        try:
            joblib.dump(
                self.to_dict(is_materialize=False),
                tmp_filepath,
                compress=3,
            )
            tmp_filepath.replace(filepath)
        finally:
            tmp_filepath.unlink(missing_ok=True)

    @classmethod
    def from_dict(cls, data: dict) -> "SnapshotSeries":
        obj = cls(
            snapshot_dict={
                float(timestamp): (
                    PseudoObservedSnapshot.from_dict(snapshot)
                    if "sim_galactic_center" in snapshot
                    else Snapshot.from_dict(snapshot)
                )
                for timestamp, snapshot in data.items()
            }
        )
        return obj

    @classmethod
    def from_pickle(cls, filepath: Path | str) -> "SnapshotSeries":
        filepath = Path(filepath)
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_joblib(cls, filepath: Path | str) -> "SnapshotSeries":
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
