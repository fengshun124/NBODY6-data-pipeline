from pathlib import Path

from nbody6.parser.base import FileParserBase, FileParserConfig

FORT83_PARSER_CONFIG = FileParserConfig(
    header_prefix="## BEGIN",
    footer_prefix="## END",
    header_length=1,
    header_schema={"time": (1, float)},
    data_schema={
        "name": (0, int),
        "x": (2, float),
        "y": (3, float),
        "z": (4, float),
        "mass": (5, float),
        "zlum": (6, float),
        "rad": (7, float),
        "tempe": (8, float),
    },
)


class Fort83Parser(FileParserBase):
    def __init__(self, path: str | Path) -> None:
        super().__init__(path, FORT83_PARSER_CONFIG)
