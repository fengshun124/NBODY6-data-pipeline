import warnings
from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from nbody6.calc.binary import calc_semi_major_axis, is_hard_binary, is_wide_binary
from nbody6.calc.cluster import calc_half_mass_radius
from nbody6.data.series import SnapshotSeries
from nbody6.data.snapshot import Snapshot
from nbody6.loader import NBody6Data
from nbody6.parser import FileBlock


class SnapshotAssembler:
    def __init__(self, raw_data: NBody6Data) -> None:
        self._raw_data = raw_data

        self._cache_assembled_dict: Dict[float, Optional[Snapshot]] = {}
        self._cache_series: Optional[SnapshotSeries] = None

    def __repr__(self) -> str:
        return f"{type(self).__name__}(raw_data={repr(self._raw_data)})"

    _POS_VEL_KEYS = ("x", "y", "z", "vx", "vy", "vz")
    _DENSITY_CENTER_DIST_KEYS = (
        "dist_dc_pc",
        "dist_dc_r_tidal",
        "dist_dc_r_half_mass",
        "is_within_r_tidal",
        "is_within_2x_r_tidal",
    )
    _ATTR_KEYS = ("mass", "zlum", "rad", "tempe")
    _BINARY_ATTR_KEYS = ("ecc", "semi", "log_period_days")
    _BINARY_PAIR_KEYS = (
        "pair",
        "obj1_name",
        "obj2_name",
        "obj1_ids",
        "obj2_ids",
        *_BINARY_ATTR_KEYS,
        "obj1_masses",
        "obj2_masses",
        "obj1_total_mass",
        "obj2_total_mass",
        "obj1_dist_dc_pc",
        "obj2_dist_dc_pc",
        "is_multi_system",
    )
    _STARS_KEYS = (
        "name",
        "is_binary",
        "is_multi_system",
        "hierarchy",
        *_POS_VEL_KEYS,
        "mass",
        "log_T_eff_K",
        "log_L_L_sol",
        "log_R_R_sol",
        *_DENSITY_CENTER_DIST_KEYS,
    )
    _BINARY_SYSTEMS_KEYS = (
        *_BINARY_PAIR_KEYS,
        "is_top_level",
        "is_wide_binary",
        "is_hard_binary",
        *_DENSITY_CENTER_DIST_KEYS,
    )

    @property
    def snapshot_series(self) -> SnapshotSeries:
        if not self._cache_series:
            raise ValueError("No cached series available. Call assemble_all() first.")
        return self._cache_series

    @property
    def cached_assembled_dict(self) -> Dict[float, Snapshot]:
        return {k: v for k, v in self._cache_assembled_dict.items() if v}

    def _build_pos_vel_df(
        self,
        o34_file_block: FileBlock,
        o9_file_block: FileBlock,
    ) -> Tuple[pd.DataFrame, Dict[int, List[int]]]:
        atomic_pos_vel_df = o34_file_block.data[["name", *self._POS_VEL_KEYS]].copy()

        reg_bin_name_map = (
            o9_file_block.data.melt(
                id_vars=["cmName"], value_vars=["name1", "name2"], value_name="name"
            )
            .drop(columns="variable")
            .groupby("cmName")["name"]
            .apply(list)
            .to_dict()
        )

        full_pos_vel_df = (
            atomic_pos_vel_df.assign(
                name=atomic_pos_vel_df["name"].map(
                    lambda x: reg_bin_name_map.get(x, [x])
                )
            )
            .explode("name")
            .loc[:, ["name", *self._POS_VEL_KEYS]]
            .astype({"name": int})
        )
        return full_pos_vel_df, reg_bin_name_map

    def _build_attr_df(
        self,
        fort82_file_block: FileBlock,
        fort83_file_block: FileBlock,
    ) -> Tuple[pd.DataFrame, Dict[int, float]]:
        reg_bin_attr_df = pd.concat(
            [
                fort82_file_block.data.rename(
                    columns={f"{attr}{i}": attr for attr in ("name", *self._ATTR_KEYS)}
                )[["name", *self._ATTR_KEYS]]
                for i in (1, 2)
            ],
            ignore_index=True,
        ).drop_duplicates(subset=["name"])

        full_attr_df = pd.concat(
            [reg_bin_attr_df, fort83_file_block.data[["name", *self._ATTR_KEYS]]],
            ignore_index=True,
        ).astype({"name": int})

        if full_attr_df["name"].duplicated().any():
            dup_names = full_attr_df.loc[
                full_attr_df["name"].duplicated(), "name"
            ].unique()
            raise ValueError(f"Duplicate names found in attributes: {dup_names}.")

        return full_attr_df, full_attr_df.set_index("name")["mass"].to_dict()

    def _build_star_df(
        self,
        timestamp: float,
        pos_vel_df: pd.DataFrame,
        attr_df: pd.DataFrame,
        dc_info_file_block: FileBlock,
        is_strict: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[int, Dict[str, float]], Dict[str, Union[int, float]]]:
        stars_df = (
            pd.merge(
                left=pos_vel_df,
                right=attr_df,
                on="name",
                how="inner",
            )
            .rename(
                columns={
                    # effective temperature in Kelvin in logarithmic scale
                    "tempe": "log_T_eff_K",
                    # luminosity in L_sun in logarithmic scale
                    "zlum": "log_L_L_sol",
                    # radius in solar radius in logarithmic scale
                    "rad": "log_R_R_sol",
                }
            )
            .sort_values(by="name")
            .reset_index(drop=True)
        )

        # validate merged star data
        if pos_vel_missing_names := list(
            set(pos_vel_df["name"]) - set(attr_df["name"])
        ):
            exception_msg = (
                f"[{timestamp} Myr] Names {pos_vel_missing_names} "
                "in OUT34/OUT9 are missing from fort.82/83."
            )
            if is_strict:
                raise ValueError(exception_msg)
            else:
                warnings.warn(exception_msg + " Dropping entries.")

        if attr_missing_names := list(set(attr_df["name"]) - set(pos_vel_df["name"])):
            exception_msg = (
                f"[{timestamp} Myr] Names {attr_missing_names} "
                "in fort.82/83 are missing from OUT34/OUT9."
            )
            if is_strict:
                raise ValueError(exception_msg)
            else:
                warnings.warn(exception_msg + " Dropping entries.")

        # calculate distance to density center
        density_center = dc_info_file_block.header["density_center"]
        stars_df["dist_dc_pc"] = np.linalg.norm(
            stars_df[["x", "y", "z"]].to_numpy() - density_center, axis=1
        )

        # calculate tidal radius and normalize distances
        r_tidal = dc_info_file_block.header["r_tidal"]

        stars_df = stars_df.assign(
            dist_dc_r_tidal=stars_df["dist_dc_pc"] / r_tidal,
        ).assign(
            is_within_r_tidal=stars_df["dist_dc_pc"] <= r_tidal,
            is_within_2x_r_tidal=stars_df["dist_dc_pc"] <= 2 * r_tidal,
        )

        # calculate half-mass radius using stars within 2 x tidal radius
        r_half_mass = float(
            calc_half_mass_radius(
                stars_df=stars_df.loc[
                    stars_df["dist_dc_r_tidal"] <= 2,
                    (*self._POS_VEL_KEYS, "mass"),
                ],
                center_coord_pc=density_center,
            )
        )
        # normalize distances by half-mass radius
        stars_df = stars_df.assign(
            dist_dc_r_half_mass=stars_df["dist_dc_pc"] / r_half_mass,
        )

        return (
            stars_df,
            stars_df.set_index("name")[list(self._DENSITY_CENTER_DIST_KEYS)].to_dict(
                orient="index"
            ),
            {
                "r_tidal": round(r_tidal, 4),
                "r_half_mass": round(r_half_mass, 4),
                "n_stars_within_r_tidal": int((stars_df["dist_dc_r_tidal"] <= 1).sum()),
                "n_stars_within_2x_r_tidal": int(
                    (stars_df["dist_dc_r_tidal"] <= 2).sum()
                ),
            },
        )

    def _build_binary_pair_df(
        self,
        timestamp: float,
        o9_file_block: FileBlock,
        f19_file_block: FileBlock,
        star_stat_dict: Dict[str, Union[int, float]],
        reg_bin_name_map: Dict[int, List[int]],
        mass_map: Dict[int, float],
        dist_dc_map: Dict[int, Dict[str, float]],
    ) -> Tuple[pd.DataFrame, Dict[str, Union[int, float]]]:
        # generate hierarchical pair name
        def _label_hierarchy(
            obj1_ids: Union[List[int], Tuple[int, ...]],
            obj2_ids: Union[List[int], Tuple[int, ...]],
        ) -> str:
            def _format_ids(ids: Union[List[int], Tuple[int, ...]]) -> str:
                return (
                    "(" + "+".join(map(str, sorted(map(int, ids)))) + ")"
                    if len(ids) != 1
                    else str(int(ids[0]))
                )

            def _sort_key(label: str) -> Tuple[int, int]:
                return (
                    (
                        1,
                        int(float(label.split("+")[0][1:])),
                    )
                    if label.startswith("(")
                    else (0, int(label))
                )

            group1 = _format_ids(obj1_ids)
            group2 = _format_ids(obj2_ids)

            if _sort_key(group1) <= _sort_key(group2):
                return f"{group1}+{group2}"
            else:
                return f"{group2}+{group1}"

        def _lookup_dist_dc_map(ids: List[int], map_key: str) -> Optional[float]:
            values = [
                dist_dc_map.get(i, {}).get(map_key) for i in ids if i in dist_dc_map
            ]
            return np.mean(values) if values else None

        def _calc_binary_attrs(data_df: pd.DataFrame, label: str) -> pd.DataFrame:
            if not data_df.empty:
                data_df["semi"] = data_df.apply(
                    lambda row: calc_semi_major_axis(
                        mass_M_sol1=row["mass1"],
                        mass_M_sol2=row["mass2"],
                        period_days=np.power(10, row["p"]),
                    ),
                    axis=1,
                )
                data_df["pair"] = data_df.apply(
                    lambda row: _label_hierarchy(
                        reg_bin_name_map.get(row["name1"], [row["name1"]]),
                        reg_bin_name_map.get(row["name2"], [row["name2"]]),
                    ),
                    axis=1,
                )

                return data_df.rename(
                    columns={
                        "p": "log_period_days",
                        "name1": "obj1_name",
                        "name2": "obj2_name",
                    }
                )
            else:
                warnings.warn(f"[{timestamp} Myr] No binary systems found in {label}.")
                return pd.DataFrame(
                    columns=["name1", "name2", "mass1", "mass2", "ecc", "p"]
                )

        # build regularized binary systems
        reg_bin_sys_df = _calc_binary_attrs(
            data_df=o9_file_block.data.copy(),
            label="OUT9",
        )
        # build non-regularized binary systems
        non_reg_bin_sys_df = _calc_binary_attrs(
            data_df=f19_file_block.data.copy(),
            label="FORT.19",
        )

        # merge both binary system dataframes
        full_bin_sys_df = pd.concat(
            [
                bin_sys_df
                for bin_sys_df in (reg_bin_sys_df, non_reg_bin_sys_df)
                if not bin_sys_df.empty
            ],
            ignore_index=True,
        )
        if full_bin_sys_df.empty:
            warnings.warn(f"[{timestamp} Myr] No binary systems found.")
            return pd.DataFrame(columns=list(self._BINARY_PAIR_KEYS)), {}

        full_bin_sys_df = (
            full_bin_sys_df.assign(
                obj1_ids=full_bin_sys_df["obj1_name"].map(
                    lambda x: tuple(reg_bin_name_map.get(x, (x,)))
                ),
                obj2_ids=full_bin_sys_df["obj2_name"].map(
                    lambda x: tuple(reg_bin_name_map.get(x, (x,)))
                ),
            )
            .assign(
                obj1_masses=lambda df: df["obj1_ids"].map(
                    lambda ids: tuple(mass_map[i] for i in ids if i in mass_map)
                ),
                obj2_masses=lambda df: df["obj2_ids"].map(
                    lambda ids: tuple(mass_map[i] for i in ids if i in mass_map)
                ),
                obj1_total_mass=lambda df: df["obj1_masses"].map(sum),
                obj2_total_mass=lambda df: df["obj2_masses"].map(sum),
            )
            .assign(
                obj1_dist_dc_pc=lambda df: df["obj1_ids"].map(
                    lambda ids: _lookup_dist_dc_map(ids, "dist_dc_pc")
                ),
                obj2_dist_dc_pc=lambda df: df["obj2_ids"].map(
                    lambda ids: _lookup_dist_dc_map(ids, "dist_dc_pc")
                ),
            )
            .assign(
                # system-level distances for the binary system itself
                dist_dc_pc=lambda df: df.apply(
                    lambda row: _lookup_dist_dc_map(
                        ids=row["obj1_ids"] + row["obj2_ids"],
                        map_key="dist_dc_pc",
                    ),
                    axis=1,
                ),
                dist_dc_r_tidal=lambda df: df.apply(
                    lambda row: _lookup_dist_dc_map(
                        ids=row["obj1_ids"] + row["obj2_ids"],
                        map_key="dist_dc_r_tidal",
                    ),
                    axis=1,
                ),
                dist_dc_r_half_mass=lambda df: df.apply(
                    lambda row: _lookup_dist_dc_map(
                        ids=row["obj1_ids"] + row["obj2_ids"],
                        map_key="dist_dc_r_half_mass",
                    ),
                    axis=1,
                ),
            )
            .assign(
                # if ALL of its components satisfy the condition, then mark True
                is_within_r_tidal=lambda df: df.apply(
                    lambda row: all(
                        dist_dc_map.get(i, {}).get("dist_dc_r_tidal", float("inf")) <= 1
                        for i in row["obj1_ids"] + row["obj2_ids"]
                        if i in dist_dc_map
                    ),
                    axis=1,
                ),
                is_within_2x_r_tidal=lambda df: df.apply(
                    lambda row: all(
                        dist_dc_map.get(i, {}).get("dist_dc_r_tidal", float("inf")) <= 2
                        for i in row["obj1_ids"] + row["obj2_ids"]
                        if i in dist_dc_map
                    ),
                    axis=1,
                ),
            )
            .assign(
                is_multi_system=lambda df: df.apply(
                    lambda row: len(row["obj1_ids"]) > 1 or len(row["obj2_ids"]) > 1,
                    axis=1,
                )
            )
            .assign(
                is_wide_binary=lambda df: df["semi"].apply(is_wide_binary),
                is_hard_binary=lambda df: df["semi"].apply(
                    partial(
                        is_hard_binary,
                        half_mass_radius_pc=star_stat_dict["r_half_mass"],
                        num_stars=star_stat_dict["n_stars_within_2x_r_tidal"],
                    ),
                ),
            )
        )

        # determine top-level binaries (not part of a larger hierarchical system)
        component_set = set(full_bin_sys_df["obj1_ids"]) | set(
            full_bin_sys_df["obj2_ids"]
        )
        bin_id_series = pd.Series(
            [
                tuple(sorted(a + b))
                for a, b in zip(
                    full_bin_sys_df["obj1_ids"], full_bin_sys_df["obj2_ids"]
                )
            ],
            index=full_bin_sys_df.index,
        )
        full_bin_sys_df["is_top_level"] = ~bin_id_series.isin(component_set)

        return (
            full_bin_sys_df[
                [
                    *self._BINARY_PAIR_KEYS,
                    *self._DENSITY_CENTER_DIST_KEYS,
                    "is_wide_binary",
                    "is_hard_binary",
                    "is_top_level",
                ]
            ],
            {
                "n_binary_system": len(full_bin_sys_df),
                "n_multi_system": int(full_bin_sys_df["is_multi_system"].sum()),
                "n_hard_binary": int(full_bin_sys_df["is_hard_binary"].sum()),
                "n_wide_binary": int(full_bin_sys_df["is_wide_binary"].sum()),
                "n_binary_systems_within_r_tidal": int(
                    full_bin_sys_df["is_within_r_tidal"].sum()
                ),
                "n_binary_systems_within_2x_r_tidal": int(
                    full_bin_sys_df["is_within_2x_r_tidal"].sum()
                ),
            },
        )

    @staticmethod
    def _build_header(
        o34_file_block: FileBlock,
        dc_info_file_block: FileBlock,
        star_stat_dict: Dict[str, Union[int, float]],
        bin_sys_stat_dict: Dict[str, Union[int, float]],
    ) -> Dict[str, Union[int, float, str, Tuple[float, float, float]]]:
        return {
            "time": o34_file_block.header["time"],
            "density_center": tuple(
                dc_info_file_block.header["density_center"].tolist()
            ),
            **star_stat_dict,
            **bin_sys_stat_dict,
            "r_tidal_OUT34": np.round(o34_file_block.header["rtide"], 4),
            "density_center_OUT34": tuple(
                np.round(o34_file_block.header["rd"], 4).tolist()
            ),
            "mass_center_OUT34": tuple(
                np.round(o34_file_block.header["rcm"], 4).tolist()
            ),
            "galactic_pos_OUT34": tuple(
                np.round(
                    o34_file_block.header["rg"] * o34_file_block.header["rbar"], 4
                ).tolist()
            ),
            "galactic_vel_OUT34": tuple(
                np.round(
                    o34_file_block.header["vg"] * o34_file_block.header["vstar"], 4
                ).tolist()
            ),
            "nzero": o34_file_block.header["nzero"],
            "plummer_mass_OUT34": o34_file_block.header["plummer_mass"],
        }

    def _assemble(
        self,
        timestamp: float,
        is_strict: bool = True,
    ) -> Optional[Snapshot]:
        file_block_dict = self._raw_data[timestamp]
        # check if tidal radius is positive
        if file_block_dict["densCentre.txt"].header["r_tidal"] <= 0:
            warnings.warn(
                f"[{timestamp} Myr] Stellar group dissolved (r_tidal <= 0). Aborting assembly."
            )
            return None

        # build position and velocity dataframe
        pos_vel_df, reg_bin_name_map = self._build_pos_vel_df(
            o34_file_block=file_block_dict["OUT34"],
            o9_file_block=file_block_dict["OUT9"],
        )
        # build attribute dataframe
        attr_df, mass_map = self._build_attr_df(
            fort82_file_block=file_block_dict["fort.82"],
            fort83_file_block=file_block_dict["fort.83"],
        )
        # merge to build star dataframe
        stars_df, dist_dc_map, star_stat_dict = self._build_star_df(
            timestamp=timestamp,
            pos_vel_df=pos_vel_df,
            attr_df=attr_df,
            dc_info_file_block=file_block_dict["densCentre.txt"],
            is_strict=is_strict,
        )
        # build binary pair dataframe
        binary_systems_df, bin_sys_stat_dict = self._build_binary_pair_df(
            timestamp=timestamp,
            o9_file_block=file_block_dict["OUT9"],
            f19_file_block=file_block_dict["fort.19"],
            star_stat_dict=star_stat_dict,
            reg_bin_name_map=reg_bin_name_map,
            mass_map=mass_map,
            dist_dc_map=dist_dc_map,
        )

        if not binary_systems_df.empty:
            full_star_names = set(stars_df["name"])
            full_binary_star_names = set(
                binary_systems_df[["obj1_ids", "obj2_ids"]].sum().sum()
            )

            if missing_names := sorted(list(full_binary_star_names - full_star_names)):
                exception_msg = (
                    f"[{timestamp} Myr] Names {missing_names} from binary pairing info (OUT9/fort.19) "
                    "are missing from the star catalog (built from OUT34/fort.82/fort.83)."
                )
                if is_strict:
                    raise ValueError(exception_msg)
                else:
                    warnings.warn(exception_msg + " Dropping affected binary entries.")
                    binary_systems_df = binary_systems_df[
                        binary_systems_df.apply(
                            lambda row: all(
                                name in full_star_names
                                for name in row["obj1_ids"] + row["obj2_ids"]
                            ),
                            axis=1,
                        )
                    ].reset_index(drop=True)

            star_pair_df = (
                binary_systems_df[["pair", "obj1_ids", "obj2_ids"]]
                .assign(all_ids=lambda df: df["obj1_ids"] + df["obj2_ids"])
                .explode("all_ids")[["pair", "all_ids"]]
                .rename(columns={"all_ids": "name"})
            )
            # map: star name -> list of all pair names containing this star
            hierarchy_map = (
                star_pair_df.groupby("name")["pair"]
                .apply(
                    lambda pairs: tuple(
                        sorted(
                            list(pairs) + [str(int(pairs.name))],
                            key=lambda p: (len(p), p),
                        )
                    )
                )
                .to_dict()
            )

            # a star is in a multi-system if its hierarchy contains more than 2 levels
            # (the star itself, a simple pair, and then a larger system).
            multi_system_map = {
                name: len(h_list) > 2 for name, h_list in hierarchy_map.items()
            }

            stars_df = stars_df.assign(
                is_binary=stars_df["name"].isin(hierarchy_map.keys()),
                is_multi_system=stars_df["name"].map(
                    lambda n: multi_system_map.get(n, False)
                ),
                hierarchy=stars_df["name"].map(
                    lambda n: hierarchy_map.get(n, (str(int(n)),))
                ),
            )
        else:
            # no binaries, all single stars
            stars_df = stars_df.assign(
                is_binary=False,
                is_multi_system=False,
                hierarchy=stars_df["name"].map(lambda n: (str(int(n)),)),
            )

        return Snapshot(
            time=timestamp,
            header=self._build_header(
                o34_file_block=file_block_dict["OUT34"],
                dc_info_file_block=file_block_dict["densCentre.txt"],
                star_stat_dict=star_stat_dict,
                bin_sys_stat_dict=bin_sys_stat_dict,
            ),
            stars=stars_df[list(self._STARS_KEYS)],
            binary_systems=binary_systems_df[list(self._BINARY_SYSTEMS_KEYS)],
        )

    def assemble_at(
        self,
        timestamps: Union[float, List[float]],
        is_strict: bool = True,
        is_verbose: bool = False,
    ) -> Dict[float, Optional[Snapshot]]:
        assembled_dict: Dict[float, Optional[Snapshot]] = {}
        for ts in (
            pbar := tqdm(
                timestamps if isinstance(timestamps, list) else [timestamps],
                disable=not is_verbose,
                dynamic_ncols=True,
                leave=False,
            )
        ):
            pbar.set_description(f"Assembling Snapshot@{ts} Myr")
            if ts not in self._raw_data.timestamps:
                warnings.warn(f"[{ts} Myr] Timestamp not found in raw data. Skipping.")
                continue

            # check cache first
            if ts in self._cache_assembled_dict:
                assembled_dict[ts] = self._cache_assembled_dict[ts]
                continue

            # assemble and cache
            try:
                snapshot = self._assemble(timestamp=ts, is_strict=is_strict)
                assembled_dict[ts] = snapshot
                self._cache_assembled_dict[ts] = snapshot

            except Exception as e:
                raise RuntimeError(f"[{ts} Myr] Error assembling snapshot: {e}") from e

        return assembled_dict

    def assemble_all(
        self,
        is_strict: bool = True,
        is_verbose: bool = True,
    ) -> SnapshotSeries:
        if self._cache_series is not None:
            return self._cache_series

        assembled_dict: Dict[float, Optional[Snapshot]] = {}
        for ts in (
            pbar := tqdm(
                self._raw_data.timestamps,
                disable=not is_verbose,
                dynamic_ncols=True,
                leave=False,
            )
        ):
            pbar.set_description(f"Assembling Snapshot@{ts} Myr")
            # check cache first
            if ts in self._cache_assembled_dict:
                assembled_dict[ts] = self._cache_assembled_dict[ts]
                continue
            # assemble and cache
            try:
                snapshot = self._assemble(timestamp=ts, is_strict=is_strict)
                self._cache_assembled_dict[ts] = snapshot
                if snapshot is None:
                    warnings.warn(
                        f"[{ts} Myr] Assembly stopped due to cluster dissolution."
                    )
                    break
                assembled_dict[ts] = snapshot

            except Exception as e:
                raise RuntimeError(f"[{ts} Myr] Error assembling snapshot: {e}") from e

        self._cache_series = SnapshotSeries(assembled_dict)
        return self._cache_series
