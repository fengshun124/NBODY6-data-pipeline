import gc
import logging
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, Union

import numpy as np
from nbody6.assembler import SnapshotAssembler
from nbody6.data import SnapshotSeries, SnapshotSeriesCollection
from nbody6.loader import NBody6DataLoader
from nbody6.observer import PseudoObserver

SIM_ROOT_BASE = Path("../../data").resolve()
OUTPUT_BASE = Path("../output")
SIM_ATTR_PATTERN = re.compile(r"Rad(\d{2})/zmet(\d{4})/M(\d)/(\d{4})")


def fetch_simulation_root(base_path: Path):
    if not (base_path := Path(base_path).resolve()).is_dir():
        raise ValueError(f"Base path {base_path} is not a directory.")

    simulations = []
    for path in base_path.rglob("*"):
        if not path.is_dir():
            continue

        path_pts = path.parts[-4:]
        if len(path_pts) != 4:
            continue

        pattern_match = SIM_ATTR_PATTERN.match("/".join(path_pts))
        if not pattern_match:
            continue

        simulations.append(
            (
                {
                    "init_gc_radius": (init_gc_radius := int(pattern_match.group(1))),
                    "init_metallicity": (init_metal := int(pattern_match.group(2))),
                    "init_mass_lv": (init_cluster_mass := int(pattern_match.group(3))),
                    "init_pos": (init_pos := int(pattern_match.group(4))),
                },
                path,
                f"Rad{init_gc_radius:02d}-zmet{init_metal:04d}-M{init_cluster_mass}-{init_pos:04d}",
            )
        )

    return sorted(simulations, key=lambda x: x[0]["init_mass_lv"])


def setup_logger(log_file) -> None:
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


def prepare(
    sim_path: Union[Path, str],
    sim_exp_label: str,
    sim_attr_dict: Dict[str, Union[int, float]],
    log_file: str,
) -> None:
    # prepare export directories
    cached_raw_series_dir = OUTPUT_BASE / "cache" / "raw"
    cached_obs_series_dir = OUTPUT_BASE / "cache" / "obs"
    log_dir = OUTPUT_BASE / "logs"
    summary_dir = OUTPUT_BASE / "summary"
    binary_annular_stats_dir = OUTPUT_BASE / "binary_stats"
    for path in [
        cached_raw_series_dir,
        cached_obs_series_dir,
        summary_dir,
        binary_annular_stats_dir,
        log_dir,
    ]:
        path.mkdir(parents=True, exist_ok=True)
    setup_logger(log_dir / log_file)

    # preparing data
    try:
        logging.debug(f"[{sim_exp_label}] Start processing {sim_path}")
        cached_raw_series = cached_raw_series_dir / f"{sim_exp_label}-raw.joblib"

        # load and assemble N-Body6 data
        if cached_raw_series.is_file():
            logging.debug(
                f"[{sim_exp_label}] Loading cached SnapshotSeries from {cached_raw_series}"
            )
            snapshot_series = SnapshotSeries.from_pickle(cached_raw_series)
        else:
            logging.debug(f"[{sim_exp_label}] No cache found, loading from {sim_path}")
            loader = NBody6DataLoader(root=sim_path)
            loader.load(is_strict=True, is_allow_timestamp_trim=True)
            logging.debug(f"[{sim_exp_label}] Loaded {loader}")

            assembler = SnapshotAssembler(raw_data=loader.simulation_data)
            snapshot_series = assembler.assemble_all(is_strict=False)
            logging.debug(f"[{sim_exp_label}] Assembled {assembler}")

            snapshot_series.to_joblib(cached_raw_series)

            logging.debug(
                f"[{sim_exp_label}] Cached SnapshotSeries to {cached_raw_series}"
            )

            del assembler, loader
            gc.collect()

        logging.info(f"[{sim_exp_label}] Loaded {snapshot_series}")
        # export raw data summary
        exp_summary_file = summary_dir / f"{sim_exp_label}-summary.csv"
        summary_df = snapshot_series.summary.copy()

        # assign initial attributes to each row
        for attr_key, attr_val in sim_attr_dict.items():
            summary_df[attr_key] = attr_val
        summary_df.to_csv(exp_summary_file, index=False)
        logging.debug(
            f"[{sim_exp_label}] Exported summary to {exp_summary_file} with {len(summary_df)} rows"
        )
        del summary_df
        gc.collect()

        # pseudo-observe the raw data
        cached_obs_series = cached_obs_series_dir / f"{sim_exp_label}-obs.joblib"
        if cached_obs_series.is_file():
            logging.debug(
                f"[{sim_exp_label}] Loading cached observed SnapshotSeries from {cached_obs_series}"
            )
            obs_series_collection = SnapshotSeries.from_pickle(cached_obs_series)
        else:
            logging.debug(
                f"[{sim_exp_label}] No cache found, pseudo-observing the raw data"
            )
            observer = PseudoObserver(snapshot_series)
            obs_series_collection = observer.observe(
                coordinates=[
                    [dist, 0, 0]
                    for dist in np.arange(50, 600, 50).tolist()
                    + np.arange(600, 1300, 100).tolist()
                ]
            )
            logging.debug(
                f"[{sim_exp_label}] Pseudo-observed {observer} to "
                f"{len(obs_series_collection.series_dict)} series"
            )

            obs_series_collection.to_joblib(cached_obs_series)
            logging.debug(
                f"[{sim_exp_label}] Cached observed SnapshotSeries to {cached_obs_series}"
            )

            binary_anns_stats_df = (
                obs_series_collection.binary_annular_statistics.copy()
            )
            exp_bin_ann_stats_file = (
                binary_annular_stats_dir / f"{sim_exp_label}-binnary_stats.csv"
            )
            binary_anns_stats_df.to_csv(exp_bin_ann_stats_file, index=False)

            del observer, snapshot_series
            gc.collect()

    except Exception as e:
        logging.error(f"Failed to process {sim_path}: {e}")
        gc.collect()


def prepare_all() -> None:
    from joblib import Parallel, delayed

    simulations = fetch_simulation_root(base_path=SIM_ROOT_BASE)

    Parallel(n_jobs=1)(
        delayed(prepare)(
            sim_path,
            sim_exp_label,
            sim_init_dict,
            "batch.log",
        )
        for sim_init_dict, sim_path, sim_exp_label in simulations
        if sim_init_dict["init_mass_lv"] in [1]
    )

    Parallel(n_jobs=2)(
        delayed(prepare)(
            sim_path,
            sim_exp_label,
            sim_init_dict,
            "batch.log",
        )
        for sim_init_dict, sim_path, sim_exp_label in simulations
        if sim_init_dict["init_mass_lv"] in [2, 3]
    )

    Parallel(n_jobs=30)(
        delayed(prepare)(
            sim_path,
            sim_exp_label,
            sim_init_dict,
            "batch.log",
        )
        for sim_init_dict, sim_path, sim_exp_label in simulations
        if sim_init_dict["init_mass_lv"] in [4, 5, 6, 7, 8]
    )


def prepare_test(sim_exp_label) -> None:
    if (
        cache_raw_series := Path(
            OUTPUT_BASE / f"./cache/raw/{sim_exp_label}-raw.joblib"
        ).resolve()
    ).is_file():
        snapshot_series = SnapshotSeries.from_pickle(cache_raw_series)
    else:
        loader = NBody6DataLoader(root=SIM_ROOT_BASE / sim_exp_label.replace("-", "/"))
        loader.load(is_strict=True)
        assembler = SnapshotAssembler(raw_data=loader.simulation_data)
        snapshot_series = assembler.assemble_all(is_strict=False)

    observer = PseudoObserver(snapshot_series)
    series_collection = observer.observe(coordinates=[[50, 0, 0], [250, 0, 0]])

    series_collection.to_pickle(f"/tmp/{sim_exp_label}-obs.pkl", enforce_overwrite=True)

    loaded_series_collection = SnapshotSeriesCollection.from_pickle(
        f"/tmp/{sim_exp_label}-obs.pkl"
    )

    summary_df = loaded_series_collection.summary.copy()
    bin_ann_stats_df = loaded_series_collection.binary_annular_statistics.copy()

    breakpoint()


if __name__ == "__main__":
    prepare_all()
