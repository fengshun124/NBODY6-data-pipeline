import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd

from nbody6.calc.summary import summarize_descriptive_stats

try:
    from nbody6.calc.cluster import Coordinate3D
except Exception:
    Coordinate3D = Tuple[float, float, float]


@dataclass(slots=True)
class Snapshot:
    time: float
    header: Dict[str, Union[int, float, str, Tuple[float, float, float]]]
    stars: pd.DataFrame
    binary_systems: pd.DataFrame

    # caches
    _cache_summary: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)
    _cache_binary_annular: Optional[pd.DataFrame] = field(
        default=None, init=False, repr=False
    )

    # parent invalidation hook (set by containers)
    _parent_invalidator: Optional[Callable[[], None]] = field(
        default=None, init=False, repr=False
    )

    # cache management
    def _clear_cache(self):
        self._cache_summary = None
        self._cache_binary_annular = None

    def _invalidate_self_and_parent(self):
        self._clear_cache()
        if self._parent_invalidator is not None:
            self._parent_invalidator()

    def __setattr__(self, name, value) -> None:
        if hasattr(self, "_cache_summary") and name in [
            "time",
            "header",
            "stars",
            "binary_systems",
        ]:
            # invalidate caches in this snapshot and upstream containers
            try:
                self._invalidate_self_and_parent()
            except Exception:
                self._clear_cache()
        object.__setattr__(self, name, value)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"time={self.time}, "
            f"num_stars={len(self.stars)}, "
            f"num_binary_systems={len(self.binary_systems)}"
            ")"
        )

    def __len__(self) -> int:
        return len(self.stars)

    def to_dict(self, is_materialize: bool = True) -> Dict:
        if is_materialize:
            return {
                "time": float(self.time),
                "header": dict(self.header),
                "stars": self.stars.to_dict(orient="records"),
                "binary_systems": self.binary_systems.to_dict(orient="records"),
            }
        else:
            return {
                "time": self.time,
                "header": self.header,
                "stars": self.stars,
                "binary_systems": self.binary_systems,
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
    def from_dict(cls, data: Dict) -> "Snapshot":
        return cls(
            time=float(data["time"]),
            header=dict(data["header"]),
            stars=pd.DataFrame(data["stars"])
            if isinstance(data["stars"], list)
            else data["stars"],
            binary_systems=pd.DataFrame(data["binary_systems"])
            if isinstance(data["binary_systems"], list)
            else data["binary_systems"],
        )

    @classmethod
    def from_pickle(cls, filepath: Union[str, Path]) -> "Snapshot":
        filepath = Path(filepath)
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_joblib(cls, filepath: Union[str, Path]) -> "Snapshot":
        filepath = Path(filepath)
        data = joblib.load(filepath)
        return cls.from_dict(data)

    # overall summary statistics
    @property
    def summary(self) -> pd.DataFrame:
        if self._cache_summary is None:
            self._cache_summary = self._summarize()
        return self._cache_summary

    def _summarize(self) -> pd.DataFrame:
        def _summarize(
            stars_df: pd.DataFrame, bin_sys_df: pd.DataFrame
        ) -> Dict[str, float]:
            stats = {
                "n_star": len(stars_df),
                "n_binaries": int(stars_df["is_binary"].sum())
                if not stars_df.empty
                else 0,
                "n_binary_systems": len(bin_sys_df),
                "total_mass": float(stars_df["mass"].sum())
                if not stars_df.empty
                else 0.0,
                **(
                    {}
                    if stars_df.empty
                    else summarize_descriptive_stats(stars_df["mass"], "mass")
                ),
            }
            if not bin_sys_df.empty:
                stats.update(
                    **{
                        k: v
                        for col in ["ecc", "semi", "log_period_days"]
                        for k, v in summarize_descriptive_stats(
                            bin_sys_df[col], col
                        ).items()
                    }
                )
            return stats

        # overall
        stats_dict = _summarize(stars_df=self.stars, bin_sys_df=self.binary_systems)

        # for stars within specific regions
        for criterion_key in ["is_within_r_tidal", "is_within_2x_r_tidal"]:
            stars_masked = (
                self.stars[self.stars[criterion_key]]
                if not self.stars.empty
                else self.stars
            )
            bins_masked = (
                self.binary_systems[self.binary_systems[criterion_key]]
                if not self.binary_systems.empty
                else self.binary_systems
            )
            criterion_stats = _summarize(stars_df=stars_masked, bin_sys_df=bins_masked)
            stats_dict.update(
                {
                    f"{criterion_key.replace('is_', '')}_{k}": v
                    for k, v in criterion_stats.items()
                }
            )

        return pd.DataFrame([stats_dict]).assign(
            r_tidal=self.header["r_tidal"], r_half_mass=self.header["r_half_mass"]
        )

    # binary annular statistics
    @property
    def annular_statistics(self) -> pd.DataFrame:
        if self._cache_binary_annular is None:
            self._cache_binary_annular = self._compute_annular_statistics()
        return self._cache_binary_annular

    def _compute_annular_statistics(self) -> pd.DataFrame:
        if self.binary_systems.empty:
            return pd.DataFrame()

        stars_df = self.stars.copy()
        bin_sys_df = self.binary_systems.copy()

        if stars_df.empty and bin_sys_df.empty:
            return pd.DataFrame()

        # mark stars that are not in any binary pairs
        binary_pairs = set(bin_sys_df["pair"])
        stars_df["is_binary"] = stars_df["hierarchy"].apply(
            lambda h: any(pair in h for pair in binary_pairs)
        )

        bin_sys_stats_policy = {
            "n_binary": ("pair", "size"),
            **{
                f"n_{bin_type}_binary": (f"is_{bin_type}_binary", "sum")
                for bin_type in ["wide", "hard", "unresolved"]
                if f"is_{bin_type}_binary" in bin_sys_df.columns
            },
        }
        star_stats_policy = {
            "n_single": ("is_binary", lambda s: (~s).sum()),
            "n_binary_star": ("is_binary", "sum"),
        }

        stats_dfs = []
        for dist_key in ["dist_dc_r_tidal", "dist_dc_r_half_mass"]:
            if dist_key not in stars_df.columns or dist_key not in bin_sys_df.columns:
                continue

            max_dist = max(
                stars_df[dist_key].max() if not stars_df.empty else 0,
                bin_sys_df[dist_key].max() if not bin_sys_df.empty else 0,
            )
            dist_bins = np.arange(0, int(np.ceil(max_dist)) + 1, 1)

            bin_sys_stats_df = (
                (
                    bin_sys_df.assign(
                        _bin=pd.cut(
                            bin_sys_df[dist_key],
                            dist_bins,
                            right=True,
                            include_lowest=True,
                        )
                    )
                    .groupby("_bin", observed=True)
                    .agg(**bin_sys_stats_policy)
                    .reset_index()
                )
                if not bin_sys_df.empty
                else pd.DataFrame()
            )

            star_stats_df = (
                stars_df.assign(
                    _bin=pd.cut(
                        stars_df[dist_key], dist_bins, right=True, include_lowest=True
                    )
                )
                .groupby("_bin", observed=True)
                .agg(**star_stats_policy)
                .reset_index()
            )

            stats_df = (
                pd.merge(bin_sys_stats_df, star_stats_df, on="_bin", how="outer")
                .assign(
                    radius=lambda df: df["_bin"].apply(lambda x: x.right).astype(int),
                    dist_key=dist_key,
                    n_star=lambda df: df["n_single"].fillna(0)
                    + df["n_binary_star"].fillna(0),
                )
                .drop("_bin", axis=1)
                .fillna(0)
            )

            count_cols = [col for col in stats_df.columns if col.startswith("n_")]
            stats_df[count_cols] = stats_df[count_cols].astype(int)
            stats_dfs.append(stats_df)

        if not stats_dfs:
            return pd.DataFrame()

        full_stats_df = pd.concat(stats_dfs, ignore_index=True)

        base_cols = [
            "dist_key",
            "radius",
            "n_star",
            "n_single",
            "n_binary_star",
            "n_binary",
        ]
        other_cols = [c for c in full_stats_df.columns if c not in base_cols]
        col_order = base_cols + sorted(other_cols)

        return (
            full_stats_df[[c for c in col_order if c in full_stats_df.columns]]
            .sort_values(["dist_key", "radius"])
            .reset_index(drop=True)
        )


@dataclass(slots=True)
class PseudoObservedSnapshot(Snapshot):
    # source data before observational processing
    sim_galactic_center: Coordinate3D
    raw_stars: pd.DataFrame
    raw_binary_systems: pd.DataFrame

    def __setattr__(self, name, value) -> None:
        if hasattr(self, "_cache_summary") and name in [
            "time",
            "header",
            "stars",
            "binary_systems",
            "sim_galactic_center",
            "raw_stars",
            "raw_binary_systems",
        ]:
            self._invalidate_self_and_parent()
        object.__setattr__(self, name, value)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"time={self.time}, "
            f"num_stars={len(self.stars)}, "
            f"num_binary_systems={len(self.binary_systems)}, "
            f"num_raw_stars={len(self.raw_stars)}, "
            f"num_raw_binary_systems={len(self.raw_binary_systems)}"
            ")"
        )

    def to_dict(self, is_materialize: bool = True) -> Dict:
        base_dict = Snapshot.to_dict(self, is_materialize=is_materialize)
        if is_materialize:
            # serialize sim_galactic_center as dict for readability
            if hasattr(self.sim_galactic_center, "x"):
                gc = {
                    "x": float(self.sim_galactic_center.x),
                    "y": float(self.sim_galactic_center.y),
                    "z": float(self.sim_galactic_center.z),
                }
            else:
                x, y, z = self.sim_galactic_center
                gc = {"x": float(x), "y": float(y), "z": float(z)}
            base_dict.update(
                {
                    "sim_galactic_center": gc,
                    "raw_stars": self.raw_stars.to_dict(orient="records"),
                    "raw_binary_systems": self.raw_binary_systems.to_dict(
                        orient="records"
                    ),
                }
            )
        else:
            base_dict.update(
                {
                    "sim_galactic_center": self.sim_galactic_center,
                    "raw_stars": self.raw_stars,
                    "raw_binary_systems": self.raw_binary_systems,
                }
            )
        return base_dict

    @classmethod
    def from_dict(cls, data: Dict) -> "PseudoObservedSnapshot":
        # sim_galactic_center may come as dict or tuple/list
        sgc = data["sim_galactic_center"]
        if isinstance(sgc, dict):
            x, y, z = float(sgc["x"]), float(sgc["y"]), float(sgc["z"])
        else:
            x, y, z = sgc
        try:
            sim_gc = Coordinate3D(x, y, z)
        except Exception:
            sim_gc = (x, y, z)

        return cls(
            time=float(data["time"]),
            header=dict(data["header"]),
            stars=pd.DataFrame(data["stars"])
            if isinstance(data["stars"], list)
            else data["stars"],
            binary_systems=pd.DataFrame(data["binary_systems"])
            if isinstance(data["binary_systems"], list)
            else data["binary_systems"],
            sim_galactic_center=sim_gc,
            raw_stars=pd.DataFrame(data["raw_stars"])
            if isinstance(data["raw_stars"], list)
            else data["raw_stars"],
            raw_binary_systems=pd.DataFrame(data["raw_binary_systems"])
            if isinstance(data["raw_binary_systems"], list)
            else data["raw_binary_systems"],
        )

    @property
    def source_snapshot(self) -> Snapshot:
        return Snapshot(
            time=self.time,
            header=self.header,
            stars=self.raw_stars,
            binary_systems=self.raw_binary_systems,
        )
