import gc
import logging
import os
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, Union

import numpy as np
from dotenv import load_dotenv
from joblib import Parallel, delayed
from nbody6.assembler import SnapshotAssembler
from nbody6.data import SnapshotSeries, SnapshotSeriesCollection
from nbody6.loader import NBody6DataLoader
from nbody6.observer import PseudoObserver

load_dotenv()

SIM_ROOT_BASE = Path(os.getenv("SIM_ROOT_BASE")).resolve()
OUTPUT_BASE = Path(os.getenv("OUTPUT_BASE")).resolve()
SIM_ATTR_PATTERN = re.compile(r"Rad(\d{2})/zmet(\d{4})/M(\d)/(\d{4})")


def setup_logger(log_file):
    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(asctime)s][%(processName)s][%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=3),
        ],
        force=True,
    )


def fetch_simulation_root(base_path: Path):
    if not (base_path := Path(base_path).resolve()).is_dir():
        raise ValueError(f"Base path {base_path} is not a directory.")

    simulations = []
    for path in base_path.rglob("*"):
        if path.is_dir() and len(path.parts[-4:]) == 4:
            if m := SIM_ATTR_PATTERN.match("/".join(path.parts[-4:])):
                simulations.append(
                    (
                        {
                            "init_gc_radius": int(m.group(1)),
                            "init_metallicity": int(m.group(2)),
                            "init_mass_lv": int(m.group(3)),
                            "init_pos": int(m.group(4)),
                        },
                        path,
                        f"Rad{int(m.group(1)):02d}-zmet{int(m.group(2)):04d}-M{int(m.group(3))}-{int(m.group(4)):04d}",
                    )
                )
    return sorted(simulations, key=lambda x: x[0]["init_mass_lv"])


def process(
    sim_path: Union[Path, str],
    sim_exp_label: str,
    sim_attr_dict: Dict[str, Union[int, float]],
    log_file: str,
):
    # prepare directories & logger
    raw_dir = OUTPUT_BASE / "cache" / "raw"
    obs_dir = OUTPUT_BASE / "cache" / "obs"
    summary_dir = OUTPUT_BASE / "summary"
    annular_stats_dir = OUTPUT_BASE / "annular_stats"
    log_dir = OUTPUT_BASE / "logs"
    for p in [raw_dir, obs_dir, summary_dir, annular_stats_dir, log_dir]:
        p.mkdir(parents=True, exist_ok=True)
    setup_logger(log_dir / log_file)

    try:
        cache_snapshot_series_joblib = raw_dir / f"{sim_exp_label}-raw.joblib"
        cache_obs_series_collection_joblib = obs_dir / f"{sim_exp_label}-obs.joblib"

        summary_file = summary_dir / f"{sim_exp_label}-summary.csv"
        annular_stats_file = annular_stats_dir / f"{sim_exp_label}-annular_stats.csv"

        # if final summary and bin_stats exist -> skip
        if summary_file.is_file() and annular_stats_file.is_file():
            logging.info(f"[{sim_exp_label}] summary_df & bin_stats_df exist. Skip.")
            return

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
                # start rom beginning
                loader = NBody6DataLoader(root=sim_path)
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

        # export summary_df & bin_stats_df
        summary_df = series_collection.summary.copy()
        annular_stats_df = series_collection.annular_statistics.copy()

        for k, v in sim_attr_dict.items():
            summary_df[k] = v
            annular_stats_df[k] = v

        summary_df.to_csv(summary_file, index=False)
        annular_stats_df.to_csv(annular_stats_file, index=False)
        logging.info(f"[{sim_exp_label}] Finished.")

    except Exception as e:
        logging.error(f"[{sim_exp_label}] Failed: {e!r}", exc_info=True)
        gc.collect()


def process_all(log_file="batch.log"):
    simulations = fetch_simulation_root(SIM_ROOT_BASE)

    def run(sim_dict, sim_path, sim_label):
        process(
            sim_path=sim_path,
            sim_exp_label=sim_label,
            sim_attr_dict=sim_dict,
            log_file=log_file,
        )

    Parallel(n_jobs=120)(
        delayed(run)(attr_dict, path, label)
        for attr_dict, path, label in simulations
        if attr_dict["init_mass_lv"] in [5, 6, 7, 8]
    )
    Parallel(n_jobs=30)(
        delayed(run)(attr_dict, path, label)
        for attr_dict, path, label in simulations
        if attr_dict["init_mass_lv"] in [2, 3, 4]
    )
    Parallel(n_jobs=15)(
        delayed(run)(attr_dict, path, label)
        for attr_dict, path, label in simulations
        if attr_dict["init_mass_lv"] in [1]
    )


if __name__ == "__main__":
    process_all(log_file="batch.log")

    # process(
    #     sim_path=SIM_ROOT_BASE / "Rad04/zmet0014/M2/0005",
    #     sim_exp_label="Rad04-zmet0014-M2-0005",
    #     sim_attr_dict={
    #         "init_gc_radius": 4,
    #         "init_metallicity": 14,
    #         "init_mass_lv": 2,
    #         "init_pos": 5,
    #     },
    #     log_file="test.log",
    # )
