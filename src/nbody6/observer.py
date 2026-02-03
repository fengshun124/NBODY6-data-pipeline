import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from nbody6.calc.binary import (
    calc_log_equivalent_radius,
    calc_photocentric,
    calc_total_log_luminosity,
    calc_total_mass,
)
from nbody6.calc.cluster import Coordinate3D, convert_to_offset_frame
from nbody6.calc.star import calc_log_effective_temperature_K
from nbody6.data.collection import SnapshotSeriesCollection
from nbody6.data.series import SnapshotSeries
from nbody6.data.snapshot import PseudoObservedSnapshot, Snapshot

Time = float
CacheKey = tuple[Coordinate3D, Time]


class PseudoObserver:
    UNRESOLVED_SEP_FACTOR = 0.6

    def __init__(self, snapshot_series: SnapshotSeries) -> None:
        self._snapshot_series: SnapshotSeries = snapshot_series

        self._cache_obs_snapshot_dict: dict[CacheKey, PseudoObservedSnapshot] = {}
        self._cache_series_collection: SnapshotSeriesCollection

    def __repr__(self) -> str:
        return f"{type(self).__name__}(raw_snapshot_series={self._snapshot_series})"

    @property
    def series_collection(self) -> SnapshotSeriesCollection:
        return self._cache_series_collection

    @staticmethod
    def _cache_key(coordinate: Coordinate3D, ts: float) -> CacheKey:
        return (tuple(coordinate), float(ts))

    @staticmethod
    def _merge_unresolved_binaries(
        star1_attr_dict: dict[str, float],
        star2_attr_dict: dict[str, float],
        header_dict: dict[str, int | float | str | tuple[float, float, float]],
    ) -> dict[str, float]:
        pos_pc = calc_photocentric(
            L_L_sol1=np.power(10, star1_attr_dict["log_L_L_sol"]),
            L_L_sol2=np.power(10, star2_attr_dict["log_L_L_sol"]),
            vec1=[star1_attr_dict["x"], star1_attr_dict["y"], star1_attr_dict["z"]],
            vec2=[star2_attr_dict["x"], star2_attr_dict["y"], star2_attr_dict["z"]],
        )
        vel_kms = calc_photocentric(
            L_L_sol1=np.power(10, star1_attr_dict["log_L_L_sol"]),
            L_L_sol2=np.power(10, star2_attr_dict["log_L_L_sol"]),
            vec1=[star1_attr_dict["vx"], star1_attr_dict["vy"], star1_attr_dict["vz"]],
            vec2=[star2_attr_dict["vx"], star2_attr_dict["vy"], star2_attr_dict["vz"]],
        )

        dist_dc_pc = np.linalg.norm(pos_pc - header_dict["density_center"])

        log_L = calc_total_log_luminosity(
            log_L_L_sol1=star1_attr_dict["log_L_L_sol"],
            log_L_L_sol2=star2_attr_dict["log_L_L_sol"],
        )
        log_R = calc_log_equivalent_radius(
            log_R_R_sol1=star1_attr_dict["log_R_R_sol"],
            log_R_R_sol2=star2_attr_dict["log_R_R_sol"],
        )
        log_T = calc_log_effective_temperature_K(log_L, log_R)

        return {
            **dict(zip(["x", "y", "z"], pos_pc)),
            **dict(zip(["vx", "vy", "vz"], vel_kms)),
            "mass": calc_total_mass(star1_attr_dict["mass"], star2_attr_dict["mass"]),
            "dist_dc_pc": dist_dc_pc,
            "dist_dc_r_tidal": dist_dc_pc / header_dict["r_tidal"],
            "dist_dc_r_half_mass": dist_dc_pc / header_dict["r_half_mass"],
            "log_L_L_sol": log_L,
            "log_R_R_sol": log_R,
            "log_T_eff_K": log_T,
            "is_binary": True,
            "is_unresolved_binary": True,
            "is_within_r_tidal": dist_dc_pc <= header_dict["r_tidal"],
            "is_within_2x_r_tidal": dist_dc_pc <= 2 * header_dict["r_tidal"],
        }

    def _merge_unresolved_systems(
        self,
        unresolved_bin_sys_df: pd.DataFrame,
        name_attr_map: dict[int | str, dict[str, float]],
        header_dict: dict[str, int | float | str | tuple[float, float, float]],
        pseudo_obs_coord: Coordinate3D,
    ) -> pd.DataFrame:
        if unresolved_bin_sys_df.empty:
            return pd.DataFrame()

        member_sets = [
            frozenset(ids1 + ids2)
            for ids1, ids2 in unresolved_bin_sys_df[["obj1_ids", "obj2_ids"]].to_numpy()
        ]

        top_level_idx = [
            idx
            for idx, members in enumerate(member_sets)
            if not any(
                members < other for j, other in enumerate(member_sets) if j != idx
            )
        ]

        component_cache: dict[tuple[int, ...], dict[str, float]] = {}

        def _fetch_attrs(obj_ids: list[int]) -> dict[str, float]:
            key = tuple(sorted(obj_ids))
            if len(key) == 1:
                return dict(name_attr_map[key[0]])
            if len(key) == 2:
                if key not in component_cache:
                    left = _fetch_attrs([key[0]])
                    right = _fetch_attrs([key[1]])
                    component_cache[key] = self._merge_unresolved_binaries(
                        star1_attr_dict=left,
                        star2_attr_dict=right,
                        header_dict=header_dict,
                    )
                return dict(component_cache[key])
            raise ValueError(f"Unsupported unresolved component size: {obj_ids}")

        records = [
            (
                lambda merged_attr, top_set, pair_name: {
                    **merged_attr,
                    "name": pair_name,
                    "hierarchy": tuple(
                        sorted(
                            set(
                                [str(obj_id) for obj_id in top_set]
                                + [
                                    unresolved_bin_sys_df.iloc[j]["pair"]
                                    for j, members in enumerate(member_sets)
                                    if members.issubset(top_set)
                                ]
                            ),
                            key=lambda p: (len(p), p),
                        )
                    ),
                    "is_multi_system": len(top_set) > 2,
                }
            )(
                self._merge_unresolved_binaries(
                    star1_attr_dict=_fetch_attrs(
                        unresolved_bin_sys_df.iloc[idx]["obj1_ids"]
                    ),
                    star2_attr_dict=_fetch_attrs(
                        unresolved_bin_sys_df.iloc[idx]["obj2_ids"]
                    ),
                    header_dict=header_dict,
                ),
                member_sets[idx],
                unresolved_bin_sys_df.iloc[idx]["pair"],
            )
            for idx in top_level_idx
        ]

        if not records:
            return pd.DataFrame()

        return convert_to_offset_frame(
            cluster_center_pc=pseudo_obs_coord,
            centered_stars_df=pd.DataFrame.from_records(records),
        ).assign(is_binary=True, is_unresolved_binary=True)

    def _observe(
        self,
        coordinate: Coordinate3D,
        snapshot: Snapshot,
        is_slim: bool = True,
    ) -> PseudoObservedSnapshot:
        # pre-filter to 2x r_tidal
        stars_df = snapshot.stars[snapshot.stars["is_within_2x_r_tidal"]].copy()
        bin_sys_df = snapshot.binary_systems.loc[
            snapshot.binary_systems["is_within_2x_r_tidal"]
        ].copy()

        # update hierarchy, is_binary, is_multi_system
        binary_pairs = set(bin_sys_df["pair"])
        stars_df["hierarchy"] = stars_df["hierarchy"].apply(
            lambda h: [p for p in h if p in binary_pairs or "+" not in p]
        )
        stars_df["is_binary"] = stars_df["hierarchy"].str.len() > 1
        stars_df["is_multi_system"] = stars_df["hierarchy"].str.len() > 2

        stars_df = convert_to_offset_frame(
            cluster_center_pc=coordinate, centered_stars_df=stars_df
        )

        # build name -> attr dict map for fast lookup
        name_attr_map: dict[int | str, dict[str, float]] = stars_df.set_index(
            "name"
        ).to_dict(orient="index")

        # singles
        single_stars_df = (
            stars_df[~stars_df["is_binary"]].copy().assign(is_unresolved_binary=False)
        )

        # resolvability and resolved set
        if not bin_sys_df.empty:
            bin_sys_df = bin_sys_df.copy()
            obs_dist_map = stars_df.set_index("name")["dist_pc"].to_dict()

            # mean of observed distances of all components
            bin_sys_df["dist_obs_pc"] = [
                float(np.mean([obs_dist_map[i] for i in (obj1_ids + obj2_ids)]))
                for obj1_ids, obj2_ids in bin_sys_df[
                    ["obj1_ids", "obj2_ids"]
                ].to_numpy()
            ]
            bin_sys_df["is_unresolved_binary_system"] = (
                bin_sys_df["semi"]
                <= bin_sys_df["dist_obs_pc"] * self.UNRESOLVED_SEP_FACTOR
            )

            # names participating in resolved systems
            resolved_star_names = set(
                bin_sys_df[~bin_sys_df["is_unresolved_binary_system"]][
                    ["obj1_ids", "obj2_ids"]
                ]
                .stack()
                .explode()
            )
        else:
            bin_sys_df["is_unresolved_binary_system"] = pd.Series(dtype=bool)
            resolved_star_names = set()

        resolved_stars_df = (
            stars_df[stars_df["name"].isin(resolved_star_names)]
            .copy()
            .assign(is_unresolved_binary=False, is_binary=True)
        )

        # unresolved systems -> merged measurements
        unresolved_bin_sys_df = bin_sys_df[
            bin_sys_df["is_unresolved_binary_system"]
        ].copy()
        unresolved_stars_df = self._merge_unresolved_systems(
            unresolved_bin_sys_df=unresolved_bin_sys_df,
            name_attr_map=name_attr_map,
            header_dict=snapshot.header,
            pseudo_obs_coord=coordinate,
        )

        return PseudoObservedSnapshot(
            time=snapshot.time,
            header=snapshot.header,
            sim_galactic_center=coordinate,
            stars=pd.concat(
                [single_stars_df, resolved_stars_df, unresolved_stars_df],
                ignore_index=True,
            ).pipe(
                lambda df: df[
                    ["name"]
                    + [c for c in df.columns if c.startswith("is_")]
                    + [c for c in df.columns if not c.startswith("is_") and c != "name"]
                ]
            ),
            binary_systems=bin_sys_df.sort_values(
                by=["obj1_ids", "obj2_ids"]
            ).reset_index(drop=True),
            raw_stars=pd.DataFrame(columns=snapshot.stars.columns)
            if is_slim
            else snapshot.stars,
            raw_binary_systems=pd.DataFrame(columns=snapshot.binary_systems.columns)
            if is_slim
            else snapshot.binary_systems,
        )

    def observe(
        self,
        coordinates: Coordinate3D | list[Coordinate3D],
        is_verbose: bool = True,
        is_slim: bool = True,
    ) -> SnapshotSeriesCollection:
        if (
            isinstance(coordinates, list)
            and len(coordinates) > 0
            and all(isinstance(c, (list, tuple)) for c in coordinates)
        ):
            coord_list: list[Coordinate3D] = [tuple(c) for c in coordinates]
        else:
            coord_list = [tuple(coordinates)]

        obs_series_dict: dict[Coordinate3D, SnapshotSeries] = {}

        for coord in (
            coord_pbar := tqdm(
                coord_list,
                disable=not is_verbose,
                dynamic_ncols=True,
                leave=False,
            )
        ):
            coord_pbar.set_description(f"Pseudo-observing at {coord}")
            obs_snapshot_dict: dict[Time, PseudoObservedSnapshot] = {}

            for ts, snapshot in (
                ts_pbar := tqdm(
                    self._snapshot_series,
                    disable=not is_verbose,
                    dynamic_ncols=True,
                    leave=False,
                )
            ):
                ts_pbar.set_description(f"Processing Snapshot@{ts:.2f}Myr")

                if (
                    key := self._cache_key(coord, ts)
                ) not in self._cache_obs_snapshot_dict:
                    self._cache_obs_snapshot_dict[key] = self._observe(
                        coord, snapshot, is_slim=is_slim
                    )
                obs_snapshot_dict[ts] = self._cache_obs_snapshot_dict[key]

            obs_series_dict[coord] = SnapshotSeries(snapshot_dict=obs_snapshot_dict)

        self._cache_series_collection = SnapshotSeriesCollection(
            series_dict=obs_series_dict
        )
        return self._cache_series_collection
