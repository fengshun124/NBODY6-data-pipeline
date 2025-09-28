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

        return pd.DataFrame([stats_dict])

    # binary annular statistics
    @property
    def binary_annular_statistics(self) -> pd.DataFrame:
        if self._cache_binary_annular is None:
            self._cache_binary_annular = self._compute_binary_annular_statistics()
        return self._cache_binary_annular

    def _compute_binary_annular_statistics(self) -> pd.DataFrame:
        if self.binary_systems.empty:
            return pd.DataFrame()

        stars_df = (
            self.stars[self.stars.get("is_within_2x_r_tidal", True)].copy()
            if not self.stars.empty
            else self.stars
        )
        bin_sys_df = self.binary_systems[
            self.binary_systems.get("is_within_2x_r_tidal", True)
        ].copy()

        agg_policy_dict = {
            "n_binary": lambda df: len(df),
            "n_wide_binary": lambda df: int(df["is_wide_binary"].sum())
            if not df.empty
            else 0,
            "n_hard_binary": lambda df: int(df["is_hard_binary"].sum())
            if not df.empty
            else 0,
        }

        stats_df_dict = {}
        for dist_key in ["dist_dc_r_tidal", "dist_dc_r_half_mass"]:
            if stars_df.empty and bin_sys_df.empty:
                stats_df_dict[dist_key] = pd.DataFrame(
                    columns=["rmin", "rmax", *agg_policy_dict.keys(), "n_singles"]
                ).assign(dist_key=dist_key)
                continue

            max_r = int(
                np.ceil(
                    max(
                        float(stars_df[dist_key].max()) if not stars_df.empty else 0.0,
                        float(bin_sys_df[dist_key].max())
                        if not bin_sys_df.empty
                        else 0.0,
                    )
                )
            )
            distance_bins = np.arange(0, max_r + 1, 1, dtype=float)

            bin_stats_df = self._annular_stats(
                distance_bins=distance_bins,
                data_df=bin_sys_df,
                dist_key=dist_key,
                agg_policy_dict=agg_policy_dict,
            )

            single_star_df = (
                stars_df[~stars_df.get("is_binary", False)]
                if not stars_df.empty
                else pd.DataFrame()
            )
            if not single_star_df.empty:
                single_stat_df = (
                    single_star_df.assign(
                        _bin=pd.cut(
                            single_star_df[dist_key],
                            bins=distance_bins,
                            right=True,
                            include_lowest=True,
                        )
                    )
                    .groupby("_bin", observed=True)
                    .size()
                    .reset_index(name="n_singles")
                    .assign(
                        rmin=lambda df: df["_bin"].apply(
                            lambda x: int(np.round(x.left))
                        ),
                        rmax=lambda df: df["_bin"].apply(
                            lambda x: int(np.round(x.right))
                        ),
                    )
                    .drop(columns=["_bin"])
                )
            else:
                single_stat_df = pd.DataFrame(columns=["rmin", "rmax", "n_singles"])

            stats_df_dict[dist_key] = (
                pd.merge(bin_stats_df, single_stat_df, on=["rmin", "rmax"], how="left")
                .fillna({"n_singles": 0})
                .astype({"n_singles": int})
                .assign(dist_key=dist_key)
                .sort_values(["rmin", "rmax"], ignore_index=True)
            )

        return pd.concat(
            list(stats_df_dict.values()), keys=stats_df_dict.keys(), ignore_index=True
        )

    @staticmethod
    def _annular_stats(
        distance_bins: np.ndarray,
        data_df: pd.DataFrame,
        dist_key: str,
        agg_policy_dict: dict,
    ) -> pd.DataFrame:
        if data_df.empty:
            # return empty with expected schema
            cols = ["rmin", "rmax", *agg_policy_dict.keys()]
            return pd.DataFrame(columns=cols)

        data_df = data_df.copy()
        data_df["_bin"] = pd.cut(
            data_df[dist_key], bins=distance_bins, right=True, include_lowest=True
        )

        results = []
        for bin_interval, df_bin in data_df.groupby("_bin", observed=True):
            row = {
                "rmin": int(np.round(bin_interval.left)),
                "rmax": int(np.round(bin_interval.right)),
            }
            for name, func in agg_policy_dict.items():
                row[name] = func(df_bin) if len(df_bin) > 0 else 0
            results.append(row)

        return pd.DataFrame(results)


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
