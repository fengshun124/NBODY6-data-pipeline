from pathlib import Path

import numpy as np
import pandas as pd

from nbody6.parser.base import (
    FileBlock,
    FileParserBase,
    FileParserConfig,
)

DENSITY_CENTER_PARSER_CONFIG = FileParserConfig(
    header_prefix="",
    header_length=1,
    header_schema={
        "time": (0, float),
        "r_tidal": (1, float),
        "density_center_x": (2, float),
        "density_center_y": (3, float),
        "density_center_z": (4, float),
        # "n_stars": (5, int),
        # "total_mass": (6, float),
    },
    data_schema={},
)


class DensityCenterParser(FileParserBase):
    def __init__(self, path: str | Path) -> None:
        super().__init__(path=path, parser_cfg=DENSITY_CENTER_PARSER_CONFIG)

    def parse(self, is_strict: bool = True) -> dict[float, FileBlock]:
        blocks = super().parse(is_strict)

        # assign data value as header for compatibility
        for timestamp, block in blocks.items():
            blocks[timestamp].data = pd.DataFrame([block.header])

        # pack density center into an np.array
        for timestamp, block in blocks.items():
            block.header["density_center"] = np.array(
                [
                    block.header.pop("density_center_x"),
                    block.header.pop("density_center_y"),
                    block.header.pop("density_center_z"),
                ],
                dtype=float,
            )

        return blocks
