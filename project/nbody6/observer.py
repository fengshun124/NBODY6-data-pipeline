from typing import Dict, List, Optional, Tuple, Union

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
CacheKey = Tuple[Coordinate3D, Time]


class PseudoObserver:
    UNRESOLVED_SEP_FACTOR = 0.6

    def __init__(self, snapshot_series: SnapshotSeries) -> None:
        self._snapshot_series: SnapshotSeries = snapshot_series

        self._cache_obs_snapshot_dict: Dict[CacheKey, PseudoObservedSnapshot] = {}
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
        star1_attr_dict: Dict[str, float],
        star2_attr_dict: Dict[str, float],
        header_dict: Dict[str, Union[int, float, str, Tuple[float, float, float]]],
    ) -> Dict[str, float]:
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
        name_attr_map: Dict[Union[int, str], Dict[str, float]],
        header_dict: Dict[str, Union[int, float, str, Tuple[float, float, float]]],
        pseudo_obs_coord: Coordinate3D,
    ) -> pd.DataFrame:
        if unresolved_bin_sys_df.empty:
            return pd.DataFrame(columns=["name"]).assign(
                is_binary=True, is_unresolved_binary=True
            )

        # Build once; memoize to avoid duplicate merges
        arr = unresolved_bin_sys_df[["pair", "obj1_ids", "obj2_ids"]].to_numpy()
        pair_map = {
            tuple(sorted(((o1 or []) + (o2 or [])))): (pair, o1, o2)
            for pair, o1, o2 in arr
        }
        memo: Dict[Tuple[int, ...], Dict[str, float]] = {}

        def resolve(ids: List[int]) -> Dict[str, float]:
            key = tuple(sorted(ids))
            if key in memo:
                return memo[key]
            if len(key) == 1:
                res = name_attr_map[key[0]]
            elif key in pair_map:
                _, left_ids, right_ids = pair_map[key]
                res = self._merge_unresolved_binaries(
                    resolve(left_ids), resolve(right_ids), header_dict
                )
            else:
                acc = name_attr_map[key[0]]
                for sid in key[1:]:
                    acc = self._merge_unresolved_binaries(
                        acc, name_attr_map[sid], header_dict
                    )
                res = acc
            memo[key] = res
            return res

        rows = [
            {
                "name": pair,
                **resolve((o1 or []) + (o2 or [])),
            }
            for pair, o1, o2 in arr
        ]

        return convert_to_offset_frame(
            cluster_center_pc=pseudo_obs_coord,
            centered_stars_df=pd.DataFrame.from_records(rows),
        ).assign(is_binary=True, is_unresolved_binary=True)

    def _observe(
        self, coordinate: Coordinate3D, snapshot: Snapshot
    ) -> PseudoObservedSnapshot:
        # pre-filter to 2x r_tidal
        stars_df = snapshot.stars.loc[snapshot.stars["is_within_2x_r_tidal"]].copy()
        bin_sys_df = snapshot.binary_systems.loc[
            snapshot.binary_systems["is_within_2x_r_tidal"]
        ].copy()

        # Keep only pairs whose ALL members are present among filtered stars
        if not bin_sys_df.empty:
            valid_names = set(stars_df["name"])
            member_ok = bin_sys_df.apply(
                lambda r: set((r["obj1_ids"] or []) + (r["obj2_ids"] or [])).issubset(
                    valid_names
                ),
                axis=1,
            )
            bin_sys_df = bin_sys_df.loc[member_ok].copy()

            # Rebuild hierarchy with "levels":
            # [star_name, ...pairs that include the star], pairs ordered from smaller to larger system
            if not bin_sys_df.empty:
                tmp = bin_sys_df.assign(
                    members=(bin_sys_df["obj1_ids"] + bin_sys_df["obj2_ids"])
                )
                tmp = tmp.assign(member_count=tmp["members"].str.len())

                # Explode by primitive members and sort by member_count to ensure levels from small→large
                exploded = tmp.explode("members")[["members", "pair", "member_count"]]
                exploded = exploded.sort_values("member_count", kind="mergesort")
                pairs_by_member: Dict[Union[int, str], List[str]] = (
                    exploded.groupby("members")["pair"].agg(list).to_dict()
                )

                # Assign hierarchy = [name] + pairs(in order)
                # If star has no pairs, hierarchy becomes [name]
                names = stars_df["name"].to_numpy()
                stars_df["hierarchy"] = [
                    [n] + pairs_by_member.get(n, []) for n in names
                ]
            else:
                # no valid pairs left → everyone is single: [name]
                stars_df["hierarchy"] = [[n] for n in stars_df["name"]]
        else:
            # bin_sys_df empty → leave only [name]
            stars_df["hierarchy"] = [
                (h if isinstance(h, list) and len(h) > 0 else [n])
                for h, n in zip(stars_df["hierarchy"], stars_df["name"])
            ]

        # Binary / multi-system membership based on levels length
        # binary: has at least one pair -> len>=2 ; multi-system: more than one pair level -> len>2
        hlen = stars_df["hierarchy"].map(len)
        stars_df["is_binary"] = (hlen >= 2).astype(bool)
        stars_df["is_in_multiple_system"] = (hlen > 2).astype(bool)

        # observer frame (distances etc. used later are computed after this transform)
        stars_df = convert_to_offset_frame(
            cluster_center_pc=coordinate, centered_stars_df=stars_df
        )

        name_attr_map: Dict[Union[int, str], Dict[str, float]] = stars_df.set_index(
            "name"
        ).to_dict(orient="index")

        # singles
        single_stars_df = (
            stars_df[~stars_df["is_binary"]].copy().assign(is_unresolved_binary=False)
        )

        # Resolvability and resolved set
        if not bin_sys_df.empty:
            obs_dist_map = stars_df.set_index("name")["dist_pc"].to_dict()
            # mean of observed distances of all components
            arr = bin_sys_df[["obj1_ids", "obj2_ids", "semi"]].to_numpy()
            obs_dist_pc = [
                float(np.mean([obs_dist_map[i] for i in (o1 + o2)]))
                for o1, o2, _ in arr
            ]
            bin_sys_df = bin_sys_df.copy()
            bin_sys_df["obs_dist_pc"] = obs_dist_pc
            bin_sys_df["is_unresolved_binary"] = (
                bin_sys_df["semi"]
                <= bin_sys_df["obs_dist_pc"] * self.UNRESOLVED_SEP_FACTOR
            )

            # names participating in resolved systems
            resolved_names = set(
                m
                for o1, o2 in bin_sys_df.loc[
                    ~bin_sys_df["is_unresolved_binary"], ["obj1_ids", "obj2_ids"]
                ].to_numpy()
                for m in (o1 + o2)
            )
        else:
            bin_sys_df["is_unresolved_binary"] = pd.Series(dtype=bool)
            resolved_names = set()

        resolved_stars_df = (
            stars_df[stars_df["name"].isin(resolved_names)]
            .copy()
            .assign(is_unresolved_binary=False, is_binary=True)
        )

        # Unresolved systems -> merged photocenters
        unresolved_bin_sys_df = bin_sys_df.loc[
            bin_sys_df["is_unresolved_binary"]
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
            raw_stars=snapshot.stars,
            raw_binary_systems=snapshot.binary_systems,
            stars=pd.concat(
                [single_stars_df, resolved_stars_df, unresolved_stars_df],
                ignore_index=True,
            ),
            binary_systems=bin_sys_df,
        )

    def observe(
        self,
        coordinates: Union[Coordinate3D, List[Coordinate3D]],
        is_verbose: bool = True,
    ) -> SnapshotSeriesCollection:
        # normalize to a list of tuple coords so keys are hashable
        if (
            isinstance(coordinates, list)
            and len(coordinates) > 0
            and isinstance(coordinates[0], (list, tuple))
        ):
            coord_list: List[Coordinate3D] = [tuple(c) for c in coordinates]  # type: ignore
        else:
            coord_list = [tuple(coordinates)]  # type: ignore

        obs_series_dict: Dict[Coordinate3D, SnapshotSeries] = {}

        for coord in (
            coord_pbar := tqdm(
                coord_list,
                disable=not is_verbose,
                dynamic_ncols=True,
                leave=False,
            )
        ):
            coord_pbar.set_description(f"Pseudo-observing at {coord}")
            obs_snapshot_dict: Dict[Time, PseudoObservedSnapshot] = {}

            for ts, snapshot in (
                ts_pbar := tqdm(
                    self._snapshot_series,
                    disable=not is_verbose,
                    dynamic_ncols=True,
                    leave=False,
                )
            ):
                ts_pbar.set_description(f"Processing Snapshot@{ts:.2f}Myr")
                key = self._cache_key(
                    coord, ts
                )  # _cache_key already tuples/float-casts
                if key not in self._cache_obs_snapshot_dict:
                    # pass the tuple coord through; _observe accepts Coordinate3D just fine
                    self._cache_obs_snapshot_dict[key] = self._observe(coord, snapshot)
                obs_snapshot_dict[ts] = self._cache_obs_snapshot_dict[key]

            # IMPORTANT: use the tuple `coord` as the dict key (hashable)
            obs_series_dict[coord] = SnapshotSeries(snapshot_dict=obs_snapshot_dict)

        self._cache_series_collection = SnapshotSeriesCollection(
            series_dict=obs_series_dict
        )
        return self._cache_series_collection
