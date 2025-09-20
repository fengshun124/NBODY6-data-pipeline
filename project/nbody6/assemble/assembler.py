import warnings
from abc import abstractmethod
from typing import Dict, List, Optional, Protocol, Tuple, Union

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from nbody6.assemble.snapshot import Snapshot, SnapshotSeries
from nbody6.load import FileBlock, SimulationData
from nbody6.utils.calc.binary import calc_semi_major_axis


class SnapshotAssemblerPluginBase(Protocol):
    def __call__(self, snapshot: Snapshot, **kwargs) -> Snapshot: ...

    @abstractmethod
    def __repr__(self) -> str:
        return NotImplemented


class SnapshotAssembler:
    _POS_VEL_KEYS = ["x", "y", "z", "vx", "vy", "vz"]
    _DENSITY_CENTER_DIST_KEYS = ["dist_dc_pc", "dist_dc_r_tidal"]
    _ATTR_KEYS = ["mass", "zlum", "rad", "tempe"]
    _BINARY_ATTR_KEYS = ["ecc", "semi", "log_period_days"]
    _BINARY_PAIR_KEYS = [
        "obj1_ids",
        "obj2_ids",
        "obj1_masses",
        "obj2_masses",
        "obj1_total_mass",
        "obj2_total_mass",
        "obj1_dist_dc_pc",
        "obj2_dist_dc_pc",
        "obj1_dist_dc_r_tidal",
        "obj2_dist_dc_r_tidal",
        *_BINARY_ATTR_KEYS,
    ]

    def __init__(
        self,
        raw_data: SimulationData,
        assembler_plugins: Optional[List[SnapshotAssemblerPluginBase]] = None,
    ) -> None:
        self._raw_data: SimulationData = raw_data
        self._assembler_plugins = assembler_plugins or []
        self._assemble_cache: Dict[float, Optional[Snapshot]] = {}

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f"raw_data={repr(self._raw_data)}, "
            f"plugins={self._assembler_plugins}"
            f")"
        )

    @staticmethod
    def _build_header(
        o34_file_block: FileBlock, dc_info_file_block: FileBlock
    ) -> Dict[str, Union[int, float, str, Tuple[float, float, float]]]:
        return {
            "time": o34_file_block.header["time"],
            "r_tidal": dc_info_file_block.header["r_tidal"],
            "density_center": tuple(
                dc_info_file_block.header["density_center"].tolist()
            ),
            "density_center_OUT34": tuple(o34_file_block.header["rd"].tolist()),
            "mass_center_OUT34": tuple(o34_file_block.header["rcm"].tolist()),
            "galactic_pos_OUT34": tuple(
                (o34_file_block.header["rg"] * o34_file_block.header["rbar"]).tolist()
            ),
            "galactic_vel_OUT34": tuple(
                (o34_file_block.header["vg"] * o34_file_block.header["vstar"]).tolist()
            ),
        }

    def _build_pos_vel_df(
        self,
        o34_file_block: FileBlock,
        o9_file_block: FileBlock,
        dc_info_file_block: FileBlock,
    ) -> Tuple[pd.DataFrame, Dict[int, List[int]], Dict[int, Dict[str, float]]]:
        atomic_pos_vel_df = o34_file_block.data[["name"] + self._POS_VEL_KEYS].copy()
        # calculate the distance to density center and its normalized value to r_tidal
        atomic_pos_vel_df["dist_dc_pc"] = np.linalg.norm(
            atomic_pos_vel_df[["x", "y", "z"]].to_numpy()
            - dc_info_file_block.header["density_center"],
            axis=1,
        )
        # normalized distance to density center by r_tidal
        atomic_pos_vel_df["dist_dc_r_tidal"] = (
            atomic_pos_vel_df["dist_dc_pc"] / dc_info_file_block.header["r_tidal"]
        )

        # construct regularized binary mapping from OUT9 data
        reg_bin_name_map = (
            o9_file_block.data.copy()
            .melt(id_vars=["cmName"], value_vars=["name1", "name2"], value_name="name")
            .drop(columns="variable")
            .groupby("cmName")["name"]
            .apply(list)
            .to_dict()
        )

        # extend position / velocity data
        full_pos_vel_df = pd.DataFrame(
            [
                (n, *r[self._POS_VEL_KEYS + self._DENSITY_CENTER_DIST_KEYS])
                for _, r in atomic_pos_vel_df.iterrows()
                for n in (
                    [r["name"]]
                    if r["name"] not in reg_bin_name_map
                    else reg_bin_name_map[r["name"]]
                )
            ],
            columns=["name"] + self._POS_VEL_KEYS + self._DENSITY_CENTER_DIST_KEYS,
        ).astype({"name": int})

        return (
            full_pos_vel_df,
            reg_bin_name_map,
            full_pos_vel_df.set_index("name")[self._DENSITY_CENTER_DIST_KEYS].to_dict(
                orient="index"
            ),
        )

    def _build_attr_df(
        self,
        fort82_file_block: FileBlock,
        fort83_file_block: FileBlock,
    ) -> Tuple[pd.DataFrame, Dict[int, float]]:
        reg_bin_attr_df = pd.concat(
            [
                fort82_file_block.data.copy().rename(
                    columns={f"{attr}{i}": attr for attr in ["name"] + self._ATTR_KEYS}
                )[["name"] + self._ATTR_KEYS]
                for i in (1, 2)
            ],
            ignore_index=True,
        ).drop_duplicates(subset=["name"])

        full_attr_df = pd.concat(
            [
                reg_bin_attr_df,
                fort83_file_block.data.copy()[["name"] + self._ATTR_KEYS],
            ],
            ignore_index=True,
        ).astype({"name": int})

        if full_attr_df["name"].duplicated().any():
            duplicate_names = full_attr_df[full_attr_df["name"].duplicated()][
                "name"
            ].unique()
            warnings.warn(
                f"[{fort82_file_block.meta['filepath'].split('/')[-1]}/{fort83_file_block.meta['filepath'].split('/')[-1]}] "
                f"Duplicate names found: {duplicate_names}. Using first occurrence."
            )
            full_attr_df = full_attr_df.drop_duplicates(subset=["name"], keep="first")

        mass_map = full_attr_df.set_index("name")["mass"].to_dict()
        return full_attr_df, mass_map

    def _build_binary_pair_df(
        self,
        o9_file_block: FileBlock,
        f19_file_block: FileBlock,
        reg_bin_name_map: Dict[int, List[int]],
        mass_map: Dict[int, float],
        dist_dc_map: Dict[int, Dict[str, float]],
    ) -> pd.DataFrame:
        # build regularized binary pair dataframe
        reg_bin_pair_df = (
            (
                o9_file_block.data.copy()
                .assign(
                    semi=lambda df: df.apply(
                        lambda r: calc_semi_major_axis(
                            mass_M_sol1=r["mass1"],
                            mass_M_sol2=r["mass2"],
                            period_days=np.power(10, r["p"]),
                        ),
                        axis=1,
                    ),
                )
                .rename(columns={"cmName": "pair", "p": "log_period_days"})
            )
            if not o9_file_block.data.empty
            else pd.DataFrame(
                columns=[
                    "name1",
                    "name2",
                    "mass1",
                    "mass2",
                    "pair",
                ]
                + self._BINARY_ATTR_KEYS
            )
        )

        # build non-regularized binary pair dataframe
        non_reg_bin_pair_df = (
            f19_file_block.data.copy()[["name1", "name2", "mass1", "mass2", "ecc", "p"]]
            .assign(
                pair=lambda df: df.apply(
                    lambda r: f"b-{int(r['name1'])}-{int(r['name2'])}"
                    if r["mass1"] >= r["mass2"]
                    else f"b-{int(r['name2'])}-{int(r['name1'])}",
                    axis=1,
                ),
                semi=lambda df: df.apply(
                    lambda r: calc_semi_major_axis(
                        mass_M_sol1=r["mass1"],
                        mass_M_sol2=r["mass2"],
                        period_days=np.power(10, r["p"]),
                    ),
                    axis=1,
                ),
            )
            .rename(columns={"p": "log_period_days"})
            if not f19_file_block.data.empty
            else pd.DataFrame(
                columns=[
                    "name1",
                    "name2",
                    "mass1",
                    "mass2",
                    "pair",
                ]
                + self._BINARY_ATTR_KEYS
            )
        )

        # construct full binary pair dataframe
        pair_dfs = [
            pair_df
            for pair_df in [reg_bin_pair_df, non_reg_bin_pair_df]
            if not pair_df.empty
        ]
        if not pair_dfs:
            full_pair_df = pd.DataFrame(columns=["name1"] + self._BINARY_PAIR_KEYS)
        else:

            def _lookup_dist_dc_map(
                id_list: List[int], map_key: str
            ) -> Optional[float]:
                values = [
                    dist_dc_map.get(i, {}).get(map_key)
                    for i in id_list
                    if i in dist_dc_map
                ]
                return np.mean(values) if values else None

            full_pair_df = pd.concat(pair_dfs, ignore_index=True)
            full_pair_df = (
                full_pair_df.assign(
                    obj1_ids=full_pair_df["name1"].apply(
                        lambda name: [
                            int(n) for n in reg_bin_name_map.get(name, [name])
                        ]
                    ),
                    obj2_ids=full_pair_df["name2"].apply(
                        lambda name: [
                            int(x) for x in reg_bin_name_map.get(name, [name])
                        ]
                    ),
                )
                .assign(
                    obj1_masses=lambda df: df["obj1_ids"].apply(
                        lambda ids: [mass_map.get(i) for i in ids if i in mass_map]
                    ),
                    obj2_masses=lambda df: df["obj2_ids"].apply(
                        lambda ids: [mass_map.get(i) for i in ids if i in mass_map]
                    ),
                )
                .assign(
                    obj1_total_mass=lambda df: df["obj1_masses"].apply(np.sum),
                    obj2_total_mass=lambda df: df["obj2_masses"].apply(np.sum),
                )
                .assign(
                    obj1_dist_dc_pc=lambda df: df["obj1_ids"].apply(
                        _lookup_dist_dc_map, map_key="dist_dc_pc"
                    ),
                    obj2_dist_dc_pc=lambda df: df["obj2_ids"].apply(
                        _lookup_dist_dc_map, map_key="dist_dc_pc"
                    ),
                    obj1_dist_dc_r_tidal=lambda df: df["obj1_ids"].apply(
                        _lookup_dist_dc_map, map_key="dist_dc_r_tidal"
                    ),
                    obj2_dist_dc_r_tidal=lambda df: df["obj2_ids"].apply(
                        _lookup_dist_dc_map, map_key="dist_dc_r_tidal"
                    ),
                )
            )
        return full_pair_df

    def assemble_all(
        self,
        is_strict: bool = True,
        is_verbose: bool = True,
    ) -> SnapshotSeries:
        assembled: Dict[float, Snapshot] = {}

        for ts in (
            pbar := tqdm(
                self._raw_data.timestamps,
                disable=not is_verbose,
                dynamic_ncols=True,
                leave=False,
            )
        ):
            pbar.set_description(f"Assembling Snapshot@{ts}Myr")
            result = self.assemble_at(
                timestamps=ts, is_strict=is_strict, is_verbose=False
            )

            if ts not in result:
                warnings.warn(f"Assembly stopped at {ts} Myr (returned None).")
                break
            assembled.update(result)

        return SnapshotSeries(root=self._raw_data.root, snapshots=assembled)

    def assemble_at(
        self,
        timestamps: Union[List[float], float],
        is_strict: bool = True,
        is_verbose: bool = False,
    ) -> Dict[float, Optional[Snapshot]]:
        timestamps = (
            [timestamps] if isinstance(timestamps, (int, float)) else timestamps
        )

        assembled_snapshot_dict = {}
        for ts in tqdm(
            timestamps,
            desc="Assembling snapshots",
            disable=not is_verbose,
            dynamic_ncols=True,
        ):
            if ts not in self._assemble_cache:
                self._assemble_cache[ts] = self._assemble(
                    timestamp=ts, is_strict=is_strict
                )
            if self._assemble_cache[ts] is not None:
                assembled_snapshot_dict[ts] = self._assemble_cache[ts]

        return assembled_snapshot_dict

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

        # collect header info
        header_dict = self._build_header(
            o34_file_block=file_block_dict["OUT34"],
            dc_info_file_block=file_block_dict["densCentre.txt"],
        )

        # collect position / velocity info
        pos_vel_df, reg_bin_name_map, dist_dc_map = self._build_pos_vel_df(
            o34_file_block=file_block_dict["OUT34"],
            o9_file_block=file_block_dict["OUT9"],
            dc_info_file_block=file_block_dict["densCentre.txt"],
        )

        # collect attribute info
        attr_df, mass_map = self._build_attr_df(
            fort82_file_block=file_block_dict["fort.82"],
            fort83_file_block=file_block_dict["fort.83"],
        )

        # merge position / velocity and attribute info
        stars_df = (
            pd.merge(
                pos_vel_df,
                attr_df,
                on="name",
                how="inner",
            )
            .sort_values(by="name")
            .reset_index(drop=True)
        )
        # validate merged star data
        if pos_vel_missing_names := list(
            set(pos_vel_df["name"]) - set(attr_df["name"])
        ):
            exception_msg = f"[{timestamp} Myr] Names {pos_vel_missing_names} in OUT34/OUT9 are missing from fort.82/83."
            if is_strict:
                raise ValueError(exception_msg)
            else:
                warnings.warn(exception_msg + " Dropping entries.")

        if attr_missing_names := list(set(attr_df["name"]) - set(pos_vel_df["name"])):
            exception_msg = f"[{timestamp} Myr] Names {attr_missing_names} in fort.82/83 are missing from OUT34/OUT9."
            if is_strict:
                raise ValueError(exception_msg)
            else:
                warnings.warn(exception_msg + " Dropping entries.")

        # collect binary system info
        bin_sys_df = (
            self._build_binary_pair_df(
                o9_file_block=file_block_dict["OUT9"],
                f19_file_block=file_block_dict["fort.19"],
                reg_bin_name_map=reg_bin_name_map,
                mass_map=mass_map,
                dist_dc_map=dist_dc_map,
            )
            .sort_values(by=["name1", "name2"])
            .reset_index(drop=True)
        )
        if not bin_sys_df.empty:
            # validate binary system data
            if bin_sys_df.isna().any().any():
                nan_rows = bin_sys_df[bin_sys_df.isna().any(axis=1)]
                exception_msg = f"[{timestamp} Myr] NaN values found in binary data for pairs: {nan_rows['pair'].tolist()}."
                if is_strict:
                    raise ValueError(exception_msg)
                else:
                    warnings.warn(exception_msg + " Dropping entries.")
                    bin_sys_df = bin_sys_df.dropna().reset_index(drop=True)

            if bin_sys_missing_names := sorted(
                set(bin_sys_df[["obj1_ids", "obj2_ids"]].sum().sum())
                - set(stars_df["name"])
            ):
                exception_msg = f"[{timestamp} Myr] Names {bin_sys_missing_names} in binary systems are missing from star data."
                if is_strict:
                    raise ValueError(exception_msg)
                else:
                    warnings.warn(exception_msg + " Dropping entries.")
                    bin_sys_df = bin_sys_df[
                        bin_sys_df["name1"].isin(stars_df["name"])
                        & bin_sys_df["name2"].isin(stars_df["name"])
                    ].reset_index(drop=True)

        # rename columns
        stars_df = stars_df.rename(
            columns={
                # effective temperature in Kelvin in logarithmic scale
                "tempe": "log_T_eff_K",
                # luminosity in L_sun in logarithmic scale
                "zlum": "log_L_L_sol",
                # radius in solar radius in logarithmic scale
                "rad": "log_R_R_sol",
            }
        ).assign(
            is_binary=lambda df: df["name"].isin(
                bin_sys_df[["obj1_ids", "obj2_ids"]].sum().sum()
            )
        )

        snapshot = Snapshot(
            time=header_dict["time"],
            header=header_dict,
            stars=stars_df,
            binary_systems=bin_sys_df,
        )

        # post process assembled snapshot
        for handler in self._assembler_plugins:
            snapshot = handler(snapshot)

        return snapshot
