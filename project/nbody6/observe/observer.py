import re
from abc import abstractmethod
from typing import Dict, Iterable, List, Optional, Protocol, Tuple, Union

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from nbody6.assemble.snapshot import Snapshot, SnapshotSeries
from nbody6.observe.snapshot import PseudoObservedSnapshot, PseudoObservedSnapshotSeries
from nbody6.utils.calc.binary import (
    calc_log_equivalent_radius,
    calc_photocentric,
    calc_total_log_luminosity,
    calc_total_mass,
)
from nbody6.utils.calc.cluster import Coord3D, convert_to_offset_frame
from nbody6.utils.calc.stellar import (
    calc_log_effective_temperature_K,
    calc_log_surface_flux_ratio,
)


class PseudoObserverPluginBase(Protocol):
    def __call__(self, snapshot: PseudoObservedSnapshot) -> PseudoObservedSnapshot: ...

    @abstractmethod
    def __repr__(self):
        return NotImplemented


class PseudoObserver:
    def __init__(
        self,
        raw_snapshots: SnapshotSeries,
        observer_plugins: Optional[List[PseudoObserverPluginBase]] = None,
    ) -> None:
        self._raw_snapshots = raw_snapshots
        self._observer_plugins = observer_plugins or []
        self._pseudo_observe_cache: Dict[
            Coord3D, Dict[float, PseudoObservedSnapshot]
        ] = {}

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f"num_raw_snapshots={len(self._raw_snapshots.snapshots)}, "
            f"plugins={self._observer_plugins!r}"
            ")"
        )

    def observe(
        self,
        pseudo_coords: Union[Coord3D, List[Coord3D]],
        is_verbose: bool = True,
    ) -> Dict[Coord3D, PseudoObservedSnapshotSeries]:
        pseudo_coords = (
            pseudo_coords if isinstance(pseudo_coords, list) else [pseudo_coords]
        )
        result: Dict[Coord3D, PseudoObservedSnapshotSeries] = {}

        for coord in (
            coord_pbar := tqdm(
                pseudo_coords,
                disable=not is_verbose,
                dynamic_ncols=True,
                position=0,
                leave=False,
            )
        ):
            coord_pbar.set_description(f"Pseudo-observing at {coord}")
            if coord not in self._pseudo_observe_cache:
                self._pseudo_observe_cache[coord] = {}

            observed_snapshots: Dict[float, PseudoObservedSnapshot] = {}
            for ts, snapshot in (
                ts_pbar := tqdm(
                    self._raw_snapshots.snapshots.items(),
                    disable=not is_verbose,
                    dynamic_ncols=True,
                    position=1,
                    leave=False,
                )
            ):
                ts_pbar.set_description(f"Processing Snapshot@{ts}Myr")
                if ts not in self._pseudo_observe_cache[coord]:
                    self._pseudo_observe_cache[coord][ts] = self._observe(
                        coord, snapshot
                    )
                observed_snapshots[ts] = self._pseudo_observe_cache[coord][ts]

            result[coord] = PseudoObservedSnapshotSeries(
                root=self._raw_snapshots.root,
                snapshots=observed_snapshots,
            )
        return result

    @staticmethod
    def _merge_unresolved_binaries(
        star1_attr: Dict[str, float],
        star2_attr: Dict[str, float],
        header_dict: Dict[str, float],
    ) -> Dict[str, float]:
        pos_pc = calc_photocentric(
            L_L_sol1=np.power(10, star1_attr["log_L_L_sol"]),
            L_L_sol2=np.power(10, star2_attr["log_L_L_sol"]),
            vec1=[star1_attr["x"], star1_attr["y"], star1_attr["z"]],
            vec2=[star2_attr["x"], star2_attr["y"], star2_attr["z"]],
        )
        vel_kms = calc_photocentric(
            L_L_sol1=np.power(10, star1_attr["log_L_L_sol"]),
            L_L_sol2=np.power(10, star2_attr["log_L_L_sol"]),
            vec1=[star1_attr["vx"], star1_attr["vy"], star1_attr["vz"]],
            vec2=[star2_attr["vx"], star2_attr["vy"], star2_attr["vz"]],
        )

        dist_dc_pc = np.linalg.norm(pos_pc - header_dict["density_center"])
        dist_dc_r_tidal = dist_dc_pc / header_dict["r_tidal"]

        log_L_L_sol = calc_total_log_luminosity(
            log_L_L_sol1=star1_attr["log_L_L_sol"],
            log_L_L_sol2=star2_attr["log_L_L_sol"],
        )
        log_R_R_sol = calc_log_equivalent_radius(
            log_R_R_sol1=star1_attr["log_R_R_sol"],
            log_R_R_sol2=star2_attr["log_R_R_sol"],
        )
        log_T_eff_K = calc_log_effective_temperature_K(
            log_L_L_sol=log_L_L_sol, log_R_R_sol=log_R_R_sol
        )
        log_F_F_sol = calc_log_surface_flux_ratio(log_T_eff_K=log_T_eff_K)

        return {
            **dict(zip(["x", "y", "z"], pos_pc)),
            **dict(zip(["vx", "vy", "vz"], vel_kms)),
            "mass": calc_total_mass(
                mass_M_sol1=star1_attr["mass"],
                mass_M_sol2=star2_attr["mass"],
            ),
            "dist_dc_pc": dist_dc_pc,
            "dist_dc_r_tidal": dist_dc_r_tidal,
            "log_L_L_sol": log_L_L_sol,
            "log_R_R_sol": log_R_R_sol,
            "log_T_eff_K": log_T_eff_K,
            "log_F_F_sol": log_F_F_sol,
            "is_binary": True,
            "is_unresolved_binary": True,
        }

    def _observe(
        self,
        coordinate: Coord3D,
        snapshot: Snapshot,
    ) -> PseudoObservedSnapshot:
        def _compose_label(obj1_ids: List[int], obj2_ids: List[int]) -> str:
            def _format(ids: Iterable[int]) -> str:
                return (
                    "(" + "+".join(map(str, sorted(ids))) + ")"
                    if len(ids) != 1
                    else f"{ids[0]}"
                )

            def _key_order(ids: List[int]) -> Tuple[int, int]:
                return len(ids), min(ids)

            left, right = sorted([list(obj1_ids), list(obj2_ids)], key=_key_order)
            fmt_left, fmt_right = _format(left), _format(right)
            return f"{fmt_left}+{fmt_right}"

        raw_stars_df = convert_to_offset_frame(
            cluster_center_pc=coordinate,
            centered_stars_df=snapshot.stars,
        )
        name_attr_map = raw_stars_df.set_index("name").to_dict(orient="index")

        # collect single stars from the converted dataframe
        single_stars_df = raw_stars_df[~raw_stars_df["is_binary"]].assign(
            is_unresolved_binary=False
        )

        bin_sys_df = snapshot.binary_systems.copy()
        bin_sys_df["obs_dist_pc"] = bin_sys_df.apply(
            lambda row: np.mean(
                raw_stars_df[
                    raw_stars_df["name"].isin(row[["obj1_ids", "obj2_ids"]].sum())
                ]["dist_pc"]
            ),
            axis=1,
        )

        # collect resolved binaries
        resolved_bin_sys_df = bin_sys_df[
            bin_sys_df["semi"] > bin_sys_df["obs_dist_pc"] * 0.6
        ]
        resolved_binaries_df = raw_stars_df[
            raw_stars_df["name"].isin(
                resolved_bin_sys_df[["obj1_ids", "obj2_ids"]].sum().sum()
            )
        ].assign(is_unresolved_binary=False)

        # collect unresolved binaries
        unresolved_bin_sys_df = bin_sys_df[
            ~bin_sys_df.index.isin(resolved_bin_sys_df.index)
        ]
        memo: Dict[Tuple[int, ...], Dict[str, float]] = {}

        def _resolve_group(ids: List[int]) -> Dict[str, float]:
            sorted_ids = tuple(sorted(ids))
            if sorted_ids in memo:
                return memo[sorted_ids]
            if len(sorted_ids) == 1:
                record = name_attr_map[sorted_ids[0]]
            elif len(sorted_ids) == 2:
                record = self._merge_unresolved_binaries(
                    star1_attr=_resolve_group([sorted_ids[0]]),
                    star2_attr=_resolve_group([sorted_ids[1]]),
                    header_dict=snapshot.header,
                )
            else:
                raise NotImplementedError(
                    f"Merging more than 2 stars ({sorted_ids}) is not implemented yet."
                )
            memo[sorted_ids] = record
            return record

        unresolved_binaries_df = (
            pd.DataFrame(
                {
                    _compose_label(
                        pair["obj1_ids"], pair["obj2_ids"]
                    ): self._merge_unresolved_binaries(
                        star1_attr=_resolve_group(pair["obj1_ids"]),
                        star2_attr=_resolve_group(pair["obj2_ids"]),
                        header_dict=snapshot.header,
                    )
                    for _, pair in unresolved_bin_sys_df.iterrows()
                }
            )
            .T.reset_index()
            .rename(columns={"index": "name"})
        )
        if not unresolved_binaries_df.empty:
            unresolved_binaries_df = convert_to_offset_frame(
                cluster_center_pc=coordinate,
                centered_stars_df=unresolved_binaries_df,
            )

        observed_snapshot = PseudoObservedSnapshot(
            simulated_galactic_center=coordinate,
            time=snapshot.time,
            header=snapshot.header,
            observation=pd.concat(
                [
                    single_stars_df,
                    resolved_binaries_df,
                    unresolved_binaries_df,
                ],
                ignore_index=True,
            )
            .sort_values(
                by="name",
                key=lambda s: s.map(
                    lambda x: (0, int(x))
                    if isinstance(x, int)
                    else (1, int(re.search(r"\d+", str(x)).group()))
                ),
            )
            .reset_index(drop=True),
            stars=snapshot.stars,
            binary_systems=snapshot.binary_systems,
        )

        for plugin in self._observer_plugins:
            observed_snapshot = plugin(observed_snapshot)

        return observed_snapshot
