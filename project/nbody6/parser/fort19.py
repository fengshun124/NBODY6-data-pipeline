from pathlib import Path
from typing import Union

from nbody6.parser.base import FileParserBase, FileParserConfig

FORT19_PARSER_CONFIG = FileParserConfig(
    header_prefix="#",
    header_length=1,
    header_schema={
        "time": (0, float),
        "npairs": (1, int),
    },
    data_schema={
        "ecc": (3, float),
        "semi": (4, float),
        "p": (5, float),
        "mass1": (6, float),
        "mass2": (7, float),
        "name1": (8, int),
        "name2": (9, int),
        # "kstar1": (10, int),
        # "kstar2": (11, int),
        "hiarch": (12, int),
    },
)


class Fort19Parser(FileParserBase):
    def __init__(self, path: Union[str, Path]) -> None:
        super().__init__(path, FORT19_PARSER_CONFIG)
