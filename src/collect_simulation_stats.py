import gc
import logging
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
    fetch_sim_root,
    setup_logger,
)


def process(
    sim_path: Path | str,
    sim_exp_label: str,
    sim_attr_dict: dict[str, int | float],
    log_file: str,
) -> None:
    # prepare directories & logger
    raw_dir = OUTPUT_BASE / "cache" / "raw"
    obs_dir = OUTPUT_BASE / "cache" / "obs"
    overall_stats_dir = OUTPUT_BASE / "overall_stats"
    annular_stats_dir = OUTPUT_BASE / "annular_stats"
    log_dir = OUTPUT_BASE / "log"
    for p in [raw_dir, obs_dir, overall_stats_dir, annular_stats_dir, log_dir]:
        p.mkdir(parents=True, exist_ok=True)
    setup_logger(log_dir / log_file)

    logging.info(f"[{sim_exp_label}] Start processing {sim_path.resolve()} ...")

    try:
        overall_stats_file = overall_stats_dir / f"{sim_exp_label}-overall_stats.csv"
        annular_stats_file = annular_stats_dir / f"{sim_exp_label}-annular_stats.csv"

        # if final overall_stats and annular_stats exist -> skip
        if overall_stats_file.is_file() and annular_stats_file.is_file():
            logging.info(
                f"[{sim_exp_label}] overall_stats_df & annular_stats_df exist. Skip."
            )
            return

        cache_snapshot_series_joblib = raw_dir / f"{sim_exp_label}-raw.joblib"
        cache_obs_series_collection_joblib = obs_dir / f"{sim_exp_label}-obs.joblib"

        # if series_collection exist -> load & export
        if cache_obs_series_collection_joblib.is_file():
            series_collection = SnapshotSeriesCollection.from_joblib(
                cache_obs_series_collection_joblib
            )
            logging.info(f"[{sim_exp_label}] Loaded {series_collection}.")

        else:
            # if series exist -> load, compute series_collection & export
            if cache_snapshot_series_joblib.is_file():
                series = SnapshotSeries.from_joblib(cache_snapshot_series_joblib)
                logging.debug(f"[{sim_exp_label}] Loading series")
            else:
                # start from beginning -> load raw data, assemble series & export
                loader = NBODY6DataLoader(root=sim_path)
                logging.debug(f"[{sim_exp_label}] Loading {loader}")
                loader.load(is_strict=True, is_allow_timestamp_trim=True)

                assembler = SnapshotAssembler(raw_data=loader.simulation_data)
                series = assembler.assemble_all(is_strict=False)

                series.to_joblib(cache_snapshot_series_joblib)
                del loader, assembler
                gc.collect()

            logging.info(f"[{sim_exp_label}] Loaded {series}.")

            observer = PseudoObserver(series)
            series_collection = observer.observe(
                coordinates=[
                    [dist, 0, 0]
                    for dist in np.arange(50, 600, 50).tolist()
                    + np.arange(600, 1300, 100).tolist()
                ],
                is_verbose=True,
            )

            series_collection.to_joblib(cache_obs_series_collection_joblib)
            logging.info(f"[{sim_exp_label}] Computed {series_collection}.")
            del series, observer
            gc.collect()

        # export overall_stats_df & annular_stats_df
        overall_stats_df = series_collection.statistics.copy()
        annular_stats_df = series_collection.annular_statistics.copy()

        for k, v in sim_attr_dict.items():
            overall_stats_df.insert(0, k, v)
            annular_stats_df.insert(0, k, v)

        overall_stats_df.to_csv(overall_stats_file, index=False)
        annular_stats_df.to_csv(annular_stats_file, index=False)
        logging.info(f"[{sim_exp_label}] Finished.")

        del series_collection, overall_stats_df, annular_stats_df
        gc.collect()

    except Exception as e:
        logging.error(f"[{sim_exp_label}] Failed: {e!r}", exc_info=True)
        gc.collect()


def process_all(log_file: str = "batch.log") -> None:
    simulations = fetch_sim_root(SIM_ROOT_BASE)

    def run(sim_dict, sim_path, sim_label):
        process(
            sim_path=sim_path,
            sim_exp_label=sim_label,
            sim_attr_dict=sim_dict,
            log_file=log_file,
        )

    Parallel(n_jobs=30)(
        delayed(run)(attr_dict, path, label)
        for attr_dict, path, label in simulations
        if attr_dict["init_mass_lv"] in [5, 6, 7, 8]
    )
    Parallel(n_jobs=12)(
        delayed(run)(attr_dict, path, label)
        for attr_dict, path, label in simulations
        if attr_dict["init_mass_lv"] in [3, 4]
    )
    Parallel(n_jobs=4)(
        delayed(run)(attr_dict, path, label)
        for attr_dict, path, label in simulations
        if attr_dict["init_mass_lv"] in [2]
    )
    Parallel(n_jobs=1)(
        delayed(run)(attr_dict, path, label)
        for attr_dict, path, label in simulations
        if attr_dict["init_mass_lv"] in [1]
    )


if __name__ == "__main__":
    process_all(log_file="batch.log")

    # process(
    #     sim_path=SIM_ROOT_BASE / "Rad12/zmet0014/M8/0509",
    #     sim_exp_label="Rad12-zmet0014-M8-0509",
    #     sim_attr_dict={
    #         "init_gc_radius": 12,
    #         "init_metallicity": 14,
    #         "init_mass_lv": 8,
    #         "init_pos": 509,
    #     },
    #     log_file="test.log",
    # )
