import gc
import logging
import pickle
import random
import re
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from joblib import Parallel, delayed

sys.path.append(str(Path(__file__).resolve().parents[1]))


SIM_ATTR_PATTERN = re.compile(r"Rad(\d{2})/zmet(\d{4})/M(\d)/(\d{4})")
DATA_BASE_PATH = Path("../../../data/NBody6/data/").resolve()
# configure log directory
LOG_FILE_DIR = Path("../../logs").resolve()
LOG_FILE_DIR.mkdir(parents=True, exist_ok=True)


def fetch_simulation_root(base: Path):
    if not (base_path := Path(base).resolve()).is_dir():
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


def extract_output(sim_path: Path, sim_exp_label: str, log_file: str) -> None:
    setup_logger(log_file)

    from nbody6.assemble import SnapshotAssembler
    from nbody6.assemble.plugin import BasicPhotometryCalculator
    from nbody6.load import SimulationDataLoader
    from nbody6.observe import PseudoObserver
    from nbody6.observe.plugin import (
        BinaryClassifier,
        HalfMassRadiusCalculator,
        TwiceTidalRadiusCut,
    )

    cache_dir = Path("../../cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        logging.debug(f"[{sim_exp_label}] Start processing {sim_path}")

        loader = SimulationDataLoader(sim_path)
        loader.load(is_verbose=True, is_allow_timestamp_trim=True)
        logging.debug(f"[{sim_exp_label}] Loaded {loader}")

        assembler = SnapshotAssembler(
            raw_data=loader.simulation_data,
            assembler_plugins=[BasicPhotometryCalculator()],
        )
        snapshots = assembler.assemble_all(is_strict=False, is_verbose=True)
        logging.debug(
            f"[{sim_exp_label}] Assembled {assembler} with {len(snapshots)} snapshots"
        )

        snapshots.to_pickle(cache_dir / f"{sim_exp_label}-raw.pkl", is_materialize=True)

        del assembler, loader
        gc.collect()

        observer = PseudoObserver(
            raw_snapshots=snapshots,
            observer_plugins=[
                TwiceTidalRadiusCut(),
                HalfMassRadiusCalculator(),
                BinaryClassifier(),
            ],
        )
        pseudo_obs_centers = [
            (dist_pc, 0, 0)
            for dist_pc in list(range(50, 700, 50)) + list(range(700, 1300, 100))
        ]

        obs_snapshot_dict = observer.observe(pseudo_obs_centers)
        logging.debug(
            f"[{sim_exp_label}] Observed {observer} with "
            f"{len(obs_snapshot_dict) * len(snapshots)} pseudo-observed snapshots"
        )
        with open(cache_dir / f"{sim_exp_label}-obs.pkl", "wb") as f:
            pickle.dump(
                {
                    coord: snapshot_series.to_dict(is_materialize=True)
                    for coord, snapshot_series in obs_snapshot_dict.items()
                },
                f,
            )

        del snapshots, observer, obs_snapshot_dict
        gc.collect()

        logging.info(f"Finished processing {sim_path}")
    except Exception as e:
        logging.error(f"Failed to process {sim_path}: {e}")
        gc.collect()
        # raise


def extract_batch():
    setup_logger(LOG_FILE_DIR / "all.log")

    simulations = fetch_simulation_root(DATA_BASE_PATH)
    random.shuffle(simulations)
    Parallel(n_jobs=120)(
        delayed(extract_output)(
            sim_path,
            sim_exp_label,
            LOG_FILE_DIR / "batch.log",
        )
        for sim_init_dict, sim_path, sim_exp_label in simulations
        if sim_init_dict["init_mass_lv"] in [2, 3, 4]
    )


def extract_single():
    sim_path = Path(DATA_BASE_PATH / "Rad08/zmet0006/M6/0003")
    extract_output(
        sim_path,
        "Rad08-zmet0006-M6-0003",
        LOG_FILE_DIR / "single.log",
    )


if __name__ == "__main__":
    extract_batch()
    # test_single()
