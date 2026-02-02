import gc
import logging
import os
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from nbody6.assembler import SnapshotAssembler
from nbody6.data import SnapshotSeries, SnapshotSeriesCollection
from nbody6.loader import NBODY6DataLoader
from nbody6.observer import PseudoObserver
from utils import (
    OUTPUT_BASE,
    SIM_ROOT_BASE,
    atomic_export_df_csv,
    fetch_sim_root,
    setup_logger,
)

logger = logging.getLogger(__name__)


def process(
    sim_path: Path | str,
    sim_exp_label: str,
    sim_attr_dict: dict[str, int | float],
    log_file: Path | str | None = None,
    is_verbose: bool = True,
) -> None:
    # setup logger
    setup_logger(
        (
            Path(log_file).resolve()
            if log_file is not None
            else (OUTPUT_BASE / "log" / "collect_simulation.log").resolve()
        )
    )

    # prepare directories
    raw_dir = OUTPUT_BASE / "cache" / "raw"
    obs_dir = OUTPUT_BASE / "cache" / "obs"
    overall_stats_dir = OUTPUT_BASE / "stats" / "overall_stats"
    annular_stats_dir = OUTPUT_BASE / "stats" / "annular_stats"
    for p in [raw_dir, obs_dir, overall_stats_dir, annular_stats_dir]:
        p.mkdir(parents=True, exist_ok=True)

    sim_path = Path(sim_path)
    logger.debug(f"[{sim_exp_label}] start processing {sim_path.resolve()} ...")

    try:
        overall_stats_file = overall_stats_dir / f"{sim_exp_label}-overall_stats.csv"
        annular_stats_file = annular_stats_dir / f"{sim_exp_label}-annular_stats.csv"
        snapshot_series_joblib = raw_dir / f"{sim_exp_label}-raw.joblib"
        obs_series_collection_joblib = obs_dir / f"{sim_exp_label}-obs.joblib"

        # if final overall_stats and annular_stats exist -> skip
        if overall_stats_file.is_file() and annular_stats_file.is_file():
            logger.info(
                f"[{sim_exp_label}] overall_stats_df & annular_stats_df exist. Skip."
            )
            return

        # if series_collection exist -> load & export stats
        if obs_series_collection_joblib.is_file():
            series_collection = SnapshotSeriesCollection.from_joblib(
                obs_series_collection_joblib
            )
            logger.info(f"[{sim_exp_label}] loaded {series_collection}.")

        else:
            # if series exist -> load, compute series_collection & export
            if snapshot_series_joblib.is_file():
                series = SnapshotSeries.from_joblib(snapshot_series_joblib)
                logger.debug(f"[{sim_exp_label}] loaded {series}.")
            else:
                # start from beginning -> load raw data, assemble series & export
                loader = NBODY6DataLoader(root=sim_path)
                logger.debug(f"[{sim_exp_label}] loading {loader} ...")
                loader.load(
                    is_strict=True,
                    is_allow_timestamp_trim=True,
                    is_verbose=is_verbose,
                )

                assembler = SnapshotAssembler(raw_data=loader.simulation_data)
                series = assembler.assemble_all(
                    is_strict=False,
                    is_verbose=is_verbose,
                )

                series.to_joblib(snapshot_series_joblib)
                logger.debug(f"[{sim_exp_label}] assembled and saved {series}.")
                del loader, assembler
                gc.collect()

            logger.info(f"[{sim_exp_label}] loaded {series}.")

            observer = PseudoObserver(series)
            # assume star cluster located along x-axis, observe from multiple distances
            series_collection = observer.observe(
                coordinates=[
                    [dist, 0, 0]
                    for dist in np.arange(50, 600, 50).tolist()
                    + np.arange(600, 1300, 100).tolist()
                ],
                is_verbose=is_verbose,
            )

            series_collection.to_joblib(obs_series_collection_joblib)
            logger.info(
                f"[{sim_exp_label}] pseudo-observed and saved {series_collection}."
            )
            del series, observer
            gc.collect()

        # export overall_stats_df & annular_stats_df
        overall_stats_df = series_collection.statistics.copy()
        annular_stats_df = series_collection.annular_statistics.copy()
        del series_collection
        gc.collect()
        # insert simulation attributes into stats dfs
        for k, v in sim_attr_dict.items():
            overall_stats_df.insert(0, k, v)
            annular_stats_df.insert(0, k, v)
        # atomic write overall_stats & annular_stats
        atomic_export_df_csv(
            df=overall_stats_df,
            target_file=overall_stats_file,
        )
        atomic_export_df_csv(
            df=annular_stats_df,
            target_file=annular_stats_file,
        )

        logger.info(f"[{sim_exp_label}] overall_stats_df & annular_stats_df saved.")

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
            del observer
        except NameError:
            pass
        try:
            del series
        except NameError:
            pass
        try:
            del series_collection
        except NameError:
            pass
        try:
            del overall_stats_df
        except NameError:
            pass
        try:
            del annular_stats_df
        except NameError:
            pass
        gc.collect()
        logger.handlers.clear()


def process_all(log_file: Path | str | None = None) -> None:
    # setup logger
    log_file = (
        Path(log_file).resolve()
        if log_file is not None
        else (OUTPUT_BASE / "log" / "batch_collect_simulation.log").resolve()
    )
    setup_logger(log_file)

    # entry PID
    logger.info(f"Batch processing started. Entry PID: {os.getpid()}")

    simulations = fetch_sim_root(SIM_ROOT_BASE)
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
    )(
        delayed(run)(attr_dict, path, label)
        for attr_dict, path, label in simulations
        if attr_dict["init_mass_lv"] in [6, 7, 8]
    ):
        pass
    gc.collect()

    for _ in Parallel(
        n_jobs=6,
        return_as="generator",
        pre_dispatch="n_jobs",
        batch_size=1,
    )(
        delayed(run)(attr_dict, path, label)
        for attr_dict, path, label in simulations
        if attr_dict["init_mass_lv"] in [4, 5]
    ):
        pass
    gc.collect()

    for _ in Parallel(
        n_jobs=4,
        return_as="generator",
        pre_dispatch="n_jobs",
        batch_size=1,
    )(
        delayed(run)(attr_dict, path, label)
        for attr_dict, path, label in simulations
        if attr_dict["init_mass_lv"] in [2, 3]
    ):
        pass
    gc.collect()

    for _ in Parallel(
        n_jobs=1,
        return_as="generator",
        pre_dispatch="n_jobs",
        batch_size=1,
    )(
        delayed(run)(attr_dict, path, label)
        for attr_dict, path, label in simulations
        if attr_dict["init_mass_lv"] in [1]
    ):
        pass

    logger.info(f"All {len(simulations)} simulations processed.")


if __name__ == "__main__":
    process_all()
    # process(
    #     sim_path=SIM_ROOT_BASE / "Rad12/zmet0014/M8/0509",
    #     sim_exp_label="Rad12-zmet0014-M8-0509",
    #     sim_attr_dict={
    #         "init_gc_radius": 12,
    #         "init_metallicity": 14,
    #         "init_mass_lv": 8,
    #         "init_pos": 509,
    #     },
    # )
