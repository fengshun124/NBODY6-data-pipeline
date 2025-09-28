from pathlib import Path
from typing import Union

import numpy as np

from nbody6.parser import FileParserBase, FileParserConfig

OUT34_PARSER_CONFIG = FileParserConfig(
    header_prefix="#",
    header_length=1,
    header_schema={
        "time": (7, float),
        "nzero": (5, int),
        "rbar": (8, float),
        "vstar": (9, float),
        "rtide": (10, float),
        "plummer_mass": (12, float),
        "rd": ([13, 14, 15], lambda x: np.array(x, dtype=float)),
        "rcm": ([16, 17, 18], lambda x: np.array(x, dtype=float)),
        "rg": ([22, 23, 24], lambda x: np.array(x, dtype=float)),
        "vg": ([25, 26, 27], lambda x: np.array(x, dtype=float)),
    },
    data_schema={
        "x": (0, float),
        "y": (1, float),
        "z": (2, float),
        "vx": (3, float),
        "vy": (4, float),
        "vz": (5, float),
        "mass": (6, float),
        "name": (7, int),
        "kstar": (8, int),
    },
)


class OUT34Parser(FileParserBase):
    def __init__(self, path: Union[str, Path]) -> None:
        super().__init__(path, OUT34_PARSER_CONFIG)
