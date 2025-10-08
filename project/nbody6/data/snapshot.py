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
    _cache_stats: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)
    _cache_bin_annular_stats: Optional[pd.DataFrame] = field(
        default=None, init=False, repr=False
    )

    # parent invalidation hook (set by containers)
    _parent_invalidator: Optional[Callable[[], None]] = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self):
        # ensure ALL entries are not NaN
        for df_name, df in [
            ("stars", self.stars),
            ("binary_systems", self.binary_systems),
        ]:
            if df.isnull().values.any():
                nan_cols = df.columns[df.isnull().any()].tolist()
                raise ValueError(
                    f"DataFrame {df_name} contains NaN values in columns: "
                    f"{', '.join(nan_cols)}."
                )

    # cache management
    def _clear_cache(self):
        self._cache_stats = None
        self._cache_bin_annular_stats = None

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

    # overall statistics
    @property
    def statistics(self) -> pd.DataFrame:
        if self._cache_stats is None:
            self._cache_stats = self._calc_stats()
        return self._cache_stats

    def _calc_stats(self) -> pd.DataFrame:
        stars_df = self.stars
        bin_sys_df = self.binary_systems

        # early return for empty data
        if stars_df.empty and bin_sys_df.empty:
            return pd.DataFrame(
                [
                    {
                        "r_tidal": self.header.get("r_tidal", np.nan),
                        "r_half_mass": self.header.get("r_half_mass", np.nan),
                    }
                ]
            )

        star_mass = (
            stars_df["mass"].to_numpy(dtype=float, copy=False)
            if not stars_df.empty
            else np.array([])
        )
        binary_star_flags = (
            stars_df["is_binary"].to_numpy(dtype=bool, copy=False)
            if not stars_df.empty
            else np.array([], dtype=bool)
        )

        # prepare distance-based masks
        mask_specs = [("", None, None)]
        for radius_label, col_name in [
            ("r_tidal", "is_within_r_tidal"),
            ("2x_r_tidal", "is_within_2x_r_tidal"),
        ]:
            if col_name in stars_df.columns and col_name in bin_sys_df.columns:
                mask_specs.append(
                    (
                        f"within_{radius_label}_",
                        stars_df[col_name].to_numpy(dtype=bool, copy=False)
                        if not stars_df.empty
                        else None,
                        bin_sys_df[col_name].to_numpy(dtype=bool, copy=False)
                        if not bin_sys_df.empty
                        else None,
                    )
                )

        bin_sys_stat_cols = [
            col
            for col in ["ecc", "semi", "log_period_days"]
            if col in bin_sys_df.columns
        ]
        bin_sys_type_cols = [
            f"is_{bin_type}_binary"
            for bin_type in ["wide", "hard", "unresolved"]
            if f"is_{bin_type}_binary" in bin_sys_df.columns
        ]

        stats_dict: Dict[str, float] = {}

        for prefix, star_mask, bin_sys_mask in mask_specs:
            # star statistics
            if not stars_df.empty:
                masked_mass = star_mass if star_mask is None else star_mass[star_mask]
                masked_is_binary = (
                    binary_star_flags
                    if star_mask is None
                    else binary_star_flags[star_mask]
                )
                n_star = len(masked_mass)

                stats_dict[f"{prefix}n_star"] = n_star
                stats_dict[f"{prefix}n_binary_star"] = int(masked_is_binary.sum())
                stats_dict[f"{prefix}total_mass"] = float(masked_mass.sum())

                if n_star:
                    stats_dict.update(
                        {
                            f"{prefix}{k}": v
                            for k, v in summarize_descriptive_stats(
                                pd.Series(masked_mass, copy=False), "mass"
                            ).items()
                        }
                    )
            else:
                stats_dict.update(
                    {
                        f"{prefix}n_star": 0,
                        f"{prefix}n_binary_star": 0,
                        f"{prefix}total_mass": 0.0,
                    }
                )

            # binary system statistics
            if not bin_sys_df.empty:
                masked_bin_sys_df = (
                    bin_sys_df if bin_sys_mask is None else bin_sys_df[bin_sys_mask]
                )
                n_bin_sys = len(masked_bin_sys_df)
                stats_dict[f"{prefix}n_binary_system"] = n_bin_sys

                if n_bin_sys:
                    stats_dict.update(
                        {
                            f"{prefix}{k}": v
                            for col in bin_sys_stat_cols
                            for k, v in summarize_descriptive_stats(
                                masked_bin_sys_df[col], col
                            ).items()
                        }
                    )

                    stats_dict.update(
                        {
                            f"{prefix}n_{col[3:-7]}_binary_system": int(
                                masked_bin_sys_df[col].sum()
                            )
                            for col in bin_sys_type_cols
                        }
                    )
                else:
                    stats_dict.update(
                        {
                            f"{prefix}n_{col[3:-7]}_binary_system": 0
                            for col in bin_sys_type_cols
                        }
                    )
            else:
                stats_dict[f"{prefix}n_binary_system"] = 0
                stats_dict.update(
                    {
                        f"{prefix}n_{col[3:-7]}_binary_system": 0
                        for col in bin_sys_type_cols
                    }
                )

        return pd.DataFrame([stats_dict]).assign(
            r_tidal=self.header.get("r_tidal", np.nan),
            r_half_mass=self.header.get("r_half_mass", np.nan),
        )

    # binary annular statistics
    @property
    def annular_statistics(self) -> pd.DataFrame:
        if self._cache_bin_annular_stats is None:
            self._cache_bin_annular_stats = self._calc_annular_stats()
        return self._cache_bin_annular_stats

    def _calc_annular_stats(self) -> pd.DataFrame:
        # return for empty data
        if self.binary_systems.empty or (
            self.stars.empty and self.binary_systems.empty
        ):
            return pd.DataFrame()

        stars_df = self.stars
        bin_sys_df = self.binary_systems

        binary_pairs = set(bin_sys_df["pair"])
        binary_star_flags = (
            stars_df["hierarchy"]
            .apply(lambda h: bool(binary_pairs & set(h)))
            .to_numpy(dtype=np.int8, copy=False)
            if not stars_df.empty
            else np.array([], dtype=np.int8)
        )

        # extract binary system type columns
        bin_sys_type_cols = [
            f"is_{bin_type}_binary"
            for bin_type in ["wide", "hard", "unresolved"]
            if f"is_{bin_type}_binary" in bin_sys_df.columns
        ]

        annular_stats_dfs = []
        for dist_col in ["dist_dc_r_tidal", "dist_dc_r_half_mass"]:
            if dist_col not in stars_df.columns or dist_col not in bin_sys_df.columns:
                continue

            # extract and prepare radius arrays
            star_radius = (
                np.ceil(np.maximum(stars_df[dist_col].to_numpy(copy=False), 0)).astype(
                    np.int32
                )
                if not stars_df.empty
                else np.array([], dtype=np.int32)
            )
            bin_sys_radius = np.ceil(
                np.maximum(bin_sys_df[dist_col].to_numpy(copy=False), 0)
            ).astype(np.int32)
            max_radius = max(
                star_radius.max() if star_radius.size else -1,
                bin_sys_radius.max() if bin_sys_radius.size else -1,
            )
            if max_radius < 0:
                continue

            radius_range = max_radius + 1

            # build annular data dictionary with vectorized operations
            annular_data = {
                "dist_key": dist_col,
                "radius": np.arange(radius_range, dtype=np.int64),
                "n_binary_system": np.bincount(
                    bin_sys_radius, minlength=radius_range
                ).astype(np.int64),
            }

            # count stars and binary stars in annuli
            if star_radius.size:
                annular_data["n_star"] = np.bincount(
                    star_radius, minlength=radius_range
                ).astype(np.int64)
                annular_data["n_binary_star"] = np.bincount(
                    star_radius, weights=binary_star_flags, minlength=radius_range
                ).astype(np.int64)
            else:
                annular_data["n_star"] = np.zeros(radius_range, dtype=np.int64)
                annular_data["n_binary_star"] = np.zeros(radius_range, dtype=np.int64)

            annular_data["n_single"] = (
                annular_data["n_star"] - annular_data["n_binary_star"]
            )

            annular_data.update(
                {
                    f"n_{col[3:-7]}_binary_system": np.bincount(
                        bin_sys_radius,
                        weights=bin_sys_df[col].to_numpy(dtype=np.int8, copy=False),
                        minlength=radius_range,
                    ).astype(np.int64)
                    for col in bin_sys_type_cols
                }
            )

            annular_df = pd.DataFrame(annular_data)
            annular_stats_dfs.append(annular_df[annular_df["n_star"] > 0])

        if not annular_stats_dfs:
            return pd.DataFrame()

        # final assembly and ordering
        annular_stats_df = pd.concat(annular_stats_dfs, ignore_index=True)
        base_cols = [
            "dist_key",
            "radius",
            "n_star",
            "n_single",
            "n_binary_star",
            "n_binary_system",
        ]
        ordered_cols = base_cols + sorted(
            col for col in annular_stats_df.columns if col not in base_cols
        )

        return (
            annular_stats_df[ordered_cols]
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
