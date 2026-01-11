import logging
import os
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


OUTPUT_BASE = Path(os.getenv("OUTPUT_BASE")).resolve()


def setup_logger(log_file: Path | str) -> None:
    log_file = Path(log_file)

    # avoid adding multiple handlers if already set up
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        return

    handlers = [logging.StreamHandler()]
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(
            RotatingFileHandler(
                filename=str(log_file),
                mode="a",
                maxBytes=5_000_000,
                backupCount=5,
            )
        )
    except Exception as e:
        print(
            f"Failed to create log file handler for {log_file}: {e!r}, using stream handler only."
        )
    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(asctime)s][%(processName)s][%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )


SIM_ROOT_BASE = Path(os.getenv("SIM_ROOT_BASE")).resolve()
# assuming all simulation runs are stored in the structure like:
# SIM_ROOT_BASE/RadXX/zmetYYYY/MZ/PPPP/
SIM_ATTR_PATTERN = re.compile(r"Rad(\d{2})/zmet(\d{4})/M(\d)/(\d{4})")


def fetch_sim_root(base_path: Path) -> list[tuple[dict[str, int], Path, str]]:
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
