import gc
import json
import logging
import os
from pathlib import Path

import click
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from nbody6.assembler import SnapshotAssembler
from nbody6.data import Snapshot, SnapshotSeries
from nbody6.loader import NBODY6DataLoader
from utils import (
    OUTPUT_BASE,
    SIM_ROOT_BASE,
    atomic_export_df_csv,
    fetch_sim_root,
    setup_logger,
)

logger = logging.getLogger(__name__)


def _calc_inclination(r: np.ndarray, v: np.ndarray, m: np.ndarray) -> float:
    # compute CoM position
    r_com = np.average(r, weights=m, axis=0)
    r_prime = r - r_com
    # calculate angular momentum vector
    L = np.sum(m[:, None] * np.cross(r_prime, v, axis=1), axis=0)

    # inclination radian in [0, \pi]
    inclination = np.arccos(L[2] / np.linalg.norm(L))
    return inclination


def _summarize_inclination(snapshot: Snapshot) -> dict[str, object]:
    star_df = snapshot.stars

    # calculate bulk velocity using stars within r_tidal
    bulk_velocity = (
        star_df.loc[star_df["is_within_r_tidal"], ["vx", "vy", "vz"]].mean().values
    )
    # remove bulk velocity
    star_df.loc[:, ["vx", "vy", "vz"]] = star_df[["vx", "vy", "vz"]] - bulk_velocity

    # data indexed by name
    kinematic_array = star_df.set_index("name")[
        ["x", "y", "z", "vx", "vy", "vz", "mass"]
    ].to_numpy()
    distance_array = star_df.set_index("name")[
        ["dist_dc_pc", "dist_dc_r_tidal"]
    ].to_numpy()
    name_to_idx = {name: idx for idx, name in enumerate(star_df["name"].values)}

    wide_bin_sys_df = snapshot.binary_systems[
        snapshot.binary_systems["is_within_2x_r_tidal"]
        & snapshot.binary_systems["is_wide_binary_system"]
        & snapshot.binary_systems["is_top_level"]
    ]

    raw_inclinations = []
    bin_sys_names = []
    bin_sys_distances_pc = []
    bin_sys_distances_r_tidal = []

    for obj1_ids, obj2_ids in zip(
        wide_bin_sys_df["obj1_ids"], wide_bin_sys_df["obj2_ids"]
    ):
        bin_sys_ids = obj1_ids + obj2_ids
        indices = [name_to_idx[i] for i in bin_sys_ids]

        # slice data only once
        r = kinematic_array[indices, 0:3]  # positions
        v = kinematic_array[indices, 3:6]  # velocities
        m = kinematic_array[indices, 6]  # masses

        inclination = _calc_inclination(r, v, m)
        raw_inclinations.append(inclination)

        bin_sys_names.append(tuple(bin_sys_ids))

        bin_sys_distances_pc.append(
            tuple(float(distance_array[idx, 0]) for idx in indices)
        )
        bin_sys_distances_r_tidal.append(
            tuple(float(distance_array[idx, 1]) for idx in indices)
        )

    valid_inclinations = np.array(
        [inc for inc in raw_inclinations if not np.isnan(inc)]
    )

    return {
        "r_tidal": float(snapshot.header["r_tidal"]),
        "n_wide_bin_sys": int(len(raw_inclinations)),
        "n_defined_wide_bin_sys": int(len(valid_inclinations)),
        "names": json.dumps(bin_sys_names),
        "dist_pc": json.dumps(bin_sys_distances_pc),
        "dist_r_tidal": json.dumps(bin_sys_distances_r_tidal),
        "radian": json.dumps(
            [float(inc) if not np.isnan(inc) else None for inc in raw_inclinations]
        ),
        "radian_mean": (
            float(np.mean(valid_inclinations)) if len(valid_inclinations) > 0 else None
        ),
        "radian_std": (
            float(np.std(valid_inclinations)) if len(valid_inclinations) > 0 else None
        ),
        "degree": json.dumps(
            [
                float(np.degrees(inc)) if not np.isnan(inc) else None
                for inc in raw_inclinations
            ]
        ),
        "degree_mean": (
            float(np.degrees(np.mean(valid_inclinations)))
            if len(valid_inclinations) > 0
            else None
        ),
        "degree_std": (
            float(np.degrees(np.std(valid_inclinations)))
            if len(valid_inclinations) > 0
            else None
        ),
    }


def process(
    sim_path: Path | str,
    sim_exp_label: str,
    sim_attr_dict: dict[str, int | float],
    log_file: Path | str | None = None,
    is_verbose: bool = False,
) -> None:
    # setup logger
    setup_logger(
        (
            Path(log_file).resolve()
            if log_file is not None
            else (OUTPUT_BASE / "log" / "collect_inclination.log").resolve()
        )
    )

    # prepare directories
    raw_dir = OUTPUT_BASE / "cache" / "raw"
    inclination_stats_dir = OUTPUT_BASE / "stats" / "inclination_stats"
    for p in [raw_dir, inclination_stats_dir]:
        p.mkdir(parents=True, exist_ok=True)

    sim_path = Path(sim_path)
    logger.debug(f"[{sim_exp_label}] start processing {sim_path.resolve()} ...")

    try:
        inclination_stats_file = (
            inclination_stats_dir / f"{sim_exp_label}-inclination_stats.csv"
        )

        snapshot_series_joblib = raw_dir / f"{sim_exp_label}-raw.joblib"

        # if final inclination_stats exist -> skip
        if inclination_stats_file.is_file():
            logger.info(f"[{sim_exp_label}] inclination_stats_df exist. Skip.")
            return

        if snapshot_series_joblib.is_file():
            series = SnapshotSeries.from_joblib(snapshot_series_joblib)
            logger.debug(f"[{sim_exp_label}] loaded {series}.")
        else:
            loader = NBODY6DataLoader(root=sim_path)
            logger.debug(f"[{sim_exp_label}] loading {loader}")
            loader.load(
                is_strict=True,
                is_allow_timestamp_trim=True,
                is_verbose=is_verbose,
            )

            assembler = SnapshotAssembler(
                raw_data=loader.simulation_data,
            )
            series = assembler.assemble_all(
                is_strict=False,
                is_verbose=is_verbose,
            )

            series.to_joblib(snapshot_series_joblib)
            logger.debug(f"[{sim_exp_label}] assembled and saved {series}.")

            del loader, assembler
            gc.collect()

        logger.info(f"[{sim_exp_label}] loaded {series}.")
        logger.debug(f"[{sim_exp_label}] calculating inclination statistics ...")

        inclination_stats_df = pd.DataFrame(
            [
                {
                    "time": time,
                    **_summarize_inclination(snapshot),
                }
                for time, snapshot in series
            ]
        )
        del series
        gc.collect()

        # append sim attributes
        for k, v in sim_attr_dict.items():
            inclination_stats_df.insert(0, k, v)
        atomic_export_df_csv(
            df=inclination_stats_df,
            target_file=inclination_stats_file,
        )
        logger.info(f"[{sim_exp_label}] inclination_stats_df saved.")

    except Exception as e:
        logger.exception(f"[{sim_exp_label}] Failed: {e!r}")
    finally:
        try:
            del loader
        except NameError:
            pass
        try:
            del assembler
        except NameError:
            pass
        try:
            del series
        except NameError:
            pass
        try:
            del inclination_stats_df
        except NameError:
            pass
        gc.collect()
        logger.handlers.clear()


@click.command()
@click.option(
    "--log-file",
    "log_file",
    type=click.Path(),
    default=None,
    help="Path to log file. If not specified, uses default location '<OUTPUT_BASE>/log/batch_collect_inclination.log'.",
)
def process_all(log_file: Path | str | None = None) -> None:
    # setup logger
    log_file = (
        Path(log_file).resolve()
        if log_file is not None
        else (OUTPUT_BASE / "log" / "batch_collect_inclination.log").resolve()
    )
    setup_logger(log_file)

    # entry PID
    logger.info(f"Batch processing started. Entry PID: {os.getpid()}")

    simulations = fetch_sim_root(
        SIM_ROOT_BASE,
        is_reverse=True,
    )
    logger.info(f"Fetched {len(simulations)} simulations from {SIM_ROOT_BASE}.")

    def run(sim_dict, sim_path, sim_label):
        process(
            sim_path=sim_path,
            sim_exp_label=sim_label,
            sim_attr_dict=sim_dict,
            log_file=log_file,
            is_verbose=False,
        )

    for _ in Parallel(
        n_jobs=30,
        return_as="generator",
        pre_dispatch="n_jobs",
        batch_size=1,
    )(delayed(run)(attr_dict, path, label) for attr_dict, path, label in simulations):
        pass

    logger.info(f"All {len(simulations)} simulations processed.")


if __name__ == "__main__":
    process_all()
    # process(
    #     # sim_path=SIM_ROOT_BASE / "Rad04/zmet0002/M5/0005",
    #     # sim_exp_label="Rad04-zmet0002-M5-0005",
    #     sim_path=SIM_ROOT_BASE / "Rad12/zmet0014/M8/0226",
    #     sim_exp_label="Rad12-zmet0014-M8-0226",
    #     sim_attr_dict={
    #         "init_gc_radius": 12,
    #         "init_metallicity": 14,
    #         "init_mass_lv": 8,
    #         "init_pos": 226,
    #     },
    # )
