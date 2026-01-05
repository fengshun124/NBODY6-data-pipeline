import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple, Union

import joblib
import pandas as pd

from nbody6.calc.cluster import Coordinate3D
from nbody6.calc.summary import summarize_timestamp_stats
from nbody6.data.series import SnapshotSeries
from nbody6.data.snapshot import Snapshot


@dataclass(slots=True)
class SnapshotSeriesCollection:
    series_dict: Dict[Coordinate3D, SnapshotSeries]

    _cache_stats: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)
    _cache_binary_annular: Optional[pd.DataFrame] = field(
        default=None, init=False, repr=False
    )

    _parent_invalidator: Optional[Callable[[], None]] = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self):
        if not self.series_dict:
            return

        self.series_dict = {tuple(k): v for k, v in self.series_dict.items()}

        series_iter = iter(self.series_dict.values())
        try:
            first_series = next(series_iter)
            ref_timestamps = set(first_series.timestamps)
        except StopIteration:
            return

        for coord, series in self.series_dict.items():
            if set(series.timestamps) != ref_timestamps:
                raise ValueError(
                    f"Series at coordinate {coord} has timestamps that differ from the reference."
                )
        self._bind_children()

    # cache management
    def _clear_cache(self):
        self._cache_stats = None
        self._cache_binary_annular = None
        if self._parent_invalidator is not None:
            self._parent_invalidator()

    def __setattr__(self, name: str, value) -> None:
        if hasattr(self, "_cache_summary") and name == "series_dict":
            try:
                object.__setattr__(self, name, value)
            finally:
                self._bind_children()
                self._clear_cache()
            return
        object.__setattr__(self, name, value)

    def _bind_children(self):
        for series in self.series_dict.values():
            try:
                series._parent_invalidator = self._clear_cache
            except Exception:
                pass

    def __len__(self) -> int:
        return len(self.series_dict)

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f"num_series={len(self.series_dict)}, "
            f"coordinates={list(self.series_dict.keys())}"
            ")"
        )

    def __iter__(
        self,
    ) -> Iterator[Tuple[Tuple[Tuple[float, float, float], float], Snapshot]]:
        for center, series in self.series_dict.items():
            for timestamp, snapshot in series:
                yield (center, timestamp), snapshot

    def __getitem__(
        self, key: Union[Tuple[float, float, float], float]
    ) -> Union[SnapshotSeries, Dict[Tuple[float, float, float], Snapshot]]:
        if isinstance(key, tuple) and len(key) == 3:
            return self.series_dict[key]
        elif isinstance(key, (int, float)):
            return {coord: series[key] for coord, series in self.series_dict.items()}
        else:
            raise TypeError(
                f"Key must be a numeric triplet for coordinates or a number for timestamp, got {key}"
            )

    @property
    def timestamps(self) -> List[float]:
        if not self.series_dict:
            return []
        return sorted(next(iter(self.series_dict.values())).timestamps)

    @property
    def timestamp_stats(self) -> Dict[str, Dict[str, Optional[float]]]:
        return summarize_timestamp_stats(self.timestamps)

    def iter_by_time(
        self,
    ) -> Iterator[Tuple[float, Dict[Tuple[float, float, float], Snapshot]]]:
        for timestamp in self.timestamps:
            snapshots_at_time = {
                coord: series[timestamp] for coord, series in self.series_dict.items()
            }
            yield (timestamp, snapshots_at_time)

    def iter_by_coordinate(
        self,
    ) -> Iterator[Tuple[Tuple[float, float, float], SnapshotSeries]]:
        for coord, series in self.series_dict.items():
            yield (coord, series)

    def to_dict(
        self, is_materialize: bool = True
    ) -> Dict[Tuple[float, float, float], Dict]:
        return {
            coord: series.to_dict(is_materialize=is_materialize)
            for coord, series in self.series_dict.items()
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
    def from_dict(cls, data: Dict) -> "SnapshotSeriesCollection":
        return cls(
            series_dict={
                coord: SnapshotSeries.from_dict(series_data)
                for coord, series_data in data.items()
            }
        )

    @classmethod
    def from_pickle(cls, filepath: Union[str, Path]) -> "SnapshotSeriesCollection":
        filepath = Path(filepath)
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_joblib(cls, filepath: Union[str, Path]) -> "SnapshotSeriesCollection":
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
        if not self.series_dict:
            return pd.DataFrame()

        stats_dfs = []
        for coord, series in self.series_dict.items():
            series_stats = series.statistics.copy()
            series_stats.insert(0, "galactic_x", coord[0])
            series_stats.insert(1, "galactic_y", coord[1])
            series_stats.insert(2, "galactic_z", coord[2])
            stats_dfs.append(series_stats)

        return pd.concat(stats_dfs, ignore_index=True)

    # annular statistics
    @property
    def annular_statistics(self) -> pd.DataFrame:
        if self._cache_binary_annular is None:
            self._cache_binary_annular = self._calc_annular_stats()
        return self._cache_binary_annular

    def _calc_annular_stats(self) -> pd.DataFrame:
        if not self.series_dict:
            return pd.DataFrame()

        annular_stats_dfs = []
        for coord, series in self.series_dict.items():
            series_annular = series.annular_statistics
            if series_annular.empty:
                continue

            series_annular_copy = series_annular.copy()
            series_annular_copy.insert(0, "galactic_x", coord[0])
            series_annular_copy.insert(1, "galactic_y", coord[1])
            series_annular_copy.insert(2, "galactic_z", coord[2])
            annular_stats_dfs.append(series_annular_copy)

        return (
            pd.concat(annular_stats_dfs, ignore_index=True)
            if annular_stats_dfs
            else pd.DataFrame()
        )
