from pathlib import Path
from typing import Union

from nbody6.load.parser.base import FileParserBase, FileParserConfig

OUT9_PARSER_CONFIG = FileParserConfig(
    header_prefix="#",
    header_length=3,
    header_schema={
        "time": (1, float),
        "npairs": (2, int),
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
        "cmName": (13, int),
    },
)


class OUT9Parser(FileParserBase):
    def __init__(self, path: Union[str, Path]) -> None:
        super().__init__(path, OUT9_PARSER_CONFIG)
