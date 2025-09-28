from pathlib import Path
from typing import Union

from nbody6.parser.base import FileParserBase, FileParserConfig

FORT82_PARSER_CONFIG = FileParserConfig(
    header_prefix="## BEGIN",
    footer_prefix="## END",
    header_length=1,
    header_schema={"time": (1, float)},
    data_schema={
        "name1": (0, int),
        "name2": (1, int),
        "x": (5, float),
        "y": (6, float),
        "z": (7, float),
        "mass1": (11, float),
        "mass2": (12, float),
        "zlum1": (13, float),
        "zlum2": (14, float),
        "rad1": (15, float),
        "rad2": (16, float),
        "tempe1": (17, float),
        "tempe2": (18, float),
    },
)


class Fort82Parser(FileParserBase):
    def __init__(self, path: Union[str, Path]) -> None:
        super().__init__(path, FORT82_PARSER_CONFIG)
