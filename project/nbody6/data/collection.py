import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple, Union

import pandas as pd

from nbody6.calc.cluster import Coordinate3D
from nbody6.calc.summary import summarize_timestamp_stats
from nbody6.data.series import SnapshotSeries
from nbody6.data.snapshot import Snapshot


@dataclass(slots=True)
class SnapshotSeriesCollection:
    series_dict: Dict[Tuple[float, float, float], SnapshotSeries]

    _cache_summary: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)
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
        self._cache_summary = None
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

    def __iter__(self) -> Iterator[Tuple[Tuple[float, float, float], SnapshotSeries]]:
        for center, series in self.series_dict.items():
            yield (center, series)

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

    # overall summary
    @property
    def summary(self) -> pd.DataFrame:
        if self._cache_summary is None:
            self._cache_summary = self._summarize()
        return self._cache_summary

    def _summarize(self) -> pd.DataFrame:
        def _summarize_series():
            for coord, series in self.series_dict.items():
                series_summary = series.summary.copy()
                series_summary["galactic_x"] = coord[0]
                series_summary["galactic_y"] = coord[1]
                series_summary["galactic_z"] = coord[2]

                coord_cols = ["galactic_x", "galactic_y", "galactic_z"]
                other_cols = [
                    col for col in series_summary.columns if col not in coord_cols
                ]
                yield series_summary[coord_cols + other_cols]

        if not self.series_dict:
            return pd.DataFrame()

        return pd.concat(_summarize_series(), ignore_index=True)

    # binary annular statistics
    @property
    def binary_annular_statistics(self) -> pd.DataFrame:
        if self._cache_binary_annular is None:
            self._cache_binary_annular = self._compute_binary_annular_statistics()
        return self._cache_binary_annular

    def _compute_binary_annular_statistics(self) -> pd.DataFrame:
        if not self.series_dict:
            return pd.DataFrame()

        results = []
        for coord, series in self.series_dict.items():
            stats = series.binary_annular_statistics
            if stats is None or stats.empty:
                continue
            stats_df = stats.copy()
            stats_df["galactic_x"], stats_df["galactic_y"], stats_df["galactic_z"] = (
                coord
            )

            stats_df = stats_df[
                (coord_cols := ["galactic_x", "galactic_y", "galactic_z"])
                + [col for col in stats_df.columns if col not in coord_cols]
            ]
            results.append(stats_df)
        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()
