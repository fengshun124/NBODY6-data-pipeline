import itertools
import warnings
from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict

import pandas as pd


class FileBlockMeta(TypedDict, total=True):
    filepath: str
    header_line_span: str
    data_line_span: str


@dataclass
class FileBlock:
    header: Dict[str, Any]
    data: pd.DataFrame

    meta: FileBlockMeta

    def __repr__(self):
        return (
            f"{type(self).__name__}(\n"
            f"    meta={self.meta},\n"
            f"    header={self.header},\n"
            f"    data.cols={self.data.columns.tolist()},\n"
            f"    data.shape={self.data.shape}\n"
            f")"
        )


Index = Union[int, List[int]]
Converter = Callable[[Any], Any]


@dataclass(frozen=True, slots=True)
class FileParserConfig:
    header_prefix: str
    header_length: int
    header_schema: Dict[str, Tuple[Index, Converter]]
    data_schema: Dict[str, Tuple[Index, Converter]]
    footer_prefix: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "header_prefix": self.header_prefix,
            "header_length": self.header_length,
            "header_schema": self.header_schema,
            "data_schema": self.data_schema,
            "footer_prefix": self.footer_prefix,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FileParserConfig":
        return cls(
            header_prefix=data["header_prefix"],
            header_length=data["header_length"],
            header_schema=data["header_schema"],
            data_schema=data["data_schema"],
            footer_prefix=data.get("footer_prefix"),
        )


Block = Tuple[Tuple[str, List[str]], List[Tuple[str, List[str]]]]


class FileParserBase(ABC):
    def __init__(self, path: Union[str, Path], parser_cfg: FileParserConfig) -> None:
        self._class_label = type(self).__name__

        self._path = Path(path).resolve()
        if not self._path.is_file():
            raise FileNotFoundError(f"[{self._path.name}] File not found.")
        self._file_label = self._path.name

        self._parse_cfg = parser_cfg
        self._validate_parse_cfg()

        self._block_dict: Optional[Dict[float, FileBlock]] = None

    def __repr__(self) -> str:
        return f"{self._class_label}(path={self._path})"

    def __len__(self) -> int:
        self._validate_loaded()
        return len(self._block_dict)

    def __getitem__(self, timestamp: float) -> FileBlock:
        self._validate_loaded()

        if timestamp not in self._block_dict:
            nearest_ts = min(
                self._block_dict.keys(), key=lambda ts: abs(ts - timestamp)
            )
            raise KeyError(
                f"[{self._file_label}] Timestamp '{timestamp}' not found. Closest: {nearest_ts}."
            )
        return self._block_dict[timestamp]

    @property
    def name(self) -> str:
        return self._class_label

    def _validate_parse_cfg(self) -> None:
        # check if header_length is positive
        if (self._parse_cfg.header_length <= 0) or (
            not isinstance(self._parse_cfg.header_length, int)
        ):
            raise ValueError(
                f"[{self._class_label}] `header_length` must be a positive integer, got {self._parse_cfg.header_length}."
            )

    def _validate_loaded(self) -> None:
        if self._block_dict is None:
            raise ValueError(
                f"[{self._file_label}] File not loaded. Call 'parse()' first."
            )
        assert self._block_dict is not None

    @property
    def path(self) -> Path:
        return self._path.absolute()

    @property
    def timestamps(self) -> List[float]:
        self._validate_loaded()
        return sorted(self._block_dict.keys())

    def update_timestamp(self, old_timestamp: float, new_timestamp: float) -> None:
        self._validate_loaded()
        if new_timestamp in self._block_dict:
            raise KeyError(
                f"[{self._file_label}] Timestamp '{new_timestamp}' already exists."
            )

        self._block_dict[new_timestamp] = self._block_dict.pop(old_timestamp)

    @property
    def data_dict(self) -> Dict[float, FileBlock]:
        self._validate_loaded()
        return self._block_dict

    @property
    def data_list(self) -> List[Tuple[float, FileBlock]]:
        self._validate_loaded()
        return list(self._block_dict.items())

    def parse(self, is_strict: bool = True) -> Dict[float, FileBlock]:
        if self._block_dict is not None:
            warn_msg = f"[{self._file_label}] File already parsed. Re-parsing will overwrite existing data."
            warnings.warn(warn_msg, UserWarning)
        self._block_dict = {}

        def _apply_row_schema(
            ln_label: str,
            ln_idx_str: str,
            ln_tokens: List[str],
            schema: Dict[str, Tuple[Index, Converter]],
            is_strict: bool = True,
        ) -> Dict[str, Any]:
            try:
                row_dict = self._apply_schema(
                    tokens=ln_tokens, schema=schema, is_strict=is_strict
                )
                return row_dict
            except ValueError as e:
                err_msg = f"[{self._file_label} {ln_label.upper()} {ln_idx_str}] Schema application failed: {e}"
                raise ValueError(err_msg) from e

        for header_row, data_rows in self._iter_block():
            header_dict = _apply_row_schema(
                ln_label="HEADER LINE",
                ln_idx_str=header_row[0],
                ln_tokens=header_row[1],
                schema=self._parse_cfg.header_schema,
                is_strict=is_strict,
            )

            data_dicts = [
                _apply_row_schema(
                    ln_label="LINE",
                    ln_idx_str=ln_num,
                    ln_tokens=ln_tokens,
                    schema=self._parse_cfg.data_schema,
                    is_strict=is_strict,
                )
                for ln_num, ln_tokens in data_rows
            ]
            data_df = pd.DataFrame.from_records(data_dicts)

            if data_df.empty and self._parse_cfg.data_schema:
                warn_msg = f"[{self._file_label} HEADER {header_row[0]}] No data rows found for this block."
                warnings.warn(warn_msg, UserWarning)

            sort_key = "name" if "name" in self._parse_cfg.data_schema else "name1"
            if sort_key in data_df.columns:
                data_df = data_df.sort_values(by=sort_key).reset_index(drop=True)

            block_meta: FileBlockMeta = {
                "filepath": str(self._path.absolute()),
                "header_line_span": header_row[0],
                "data_line_span": (
                    f"{data_rows[0][0]}-{data_rows[-1][0]}" if data_rows else "N/A"
                ),
            }

            timestamp = round(header_dict["time"], 2)

            if timestamp in self._block_dict:
                warn_msg = (
                    f"[{self._file_label} HEADER {block_meta['header_line_span']}] "
                    f"Duplicate timestamp '{timestamp}' found. "
                    f"Overwriting block from [HEADER {self._block_dict[timestamp].meta['header_line_span']}]."
                )
                warnings.warn(warn_msg, UserWarning)

            self._block_dict[timestamp] = FileBlock(
                header=header_dict,
                data=data_df,
                meta=block_meta,
            )

        if self._block_dict:
            self._block_dict = {
                timestamp: self._block_dict[timestamp]
                for timestamp in sorted(self._block_dict.keys())
            }
        else:
            warn_msg = f"[{self._file_label}] No valid data blocks found."
            warnings.warn(warn_msg, UserWarning)
        return self._block_dict

    def _iter_block(self) -> Iterator[Block]:
        def _is_header(ln_txt: str) -> bool:
            return ln_txt.startswith(self._parse_cfg.header_prefix)

        def _is_footer(ln_txt: str) -> bool:
            footer_prefix = self._parse_cfg.footer_prefix
            return footer_prefix is not None and ln_txt.startswith(footer_prefix)

        def _yield_block(
            h_rows: List[Tuple[str, List[str]]], d_rows: List[Tuple[str, List[str]]]
        ) -> Block:
            h_ln_start, h_ln_stop = h_rows[0][0], h_rows[-1][0]
            header_ln_span = (
                f"{h_ln_start}-{h_ln_stop}"
                if h_ln_start != h_ln_stop
                else f"{h_ln_start}"
            )
            return (
                (header_ln_span, [token for _, tokens in h_rows for token in tokens]),
                [(str(ln_num), tokens) for ln_num, tokens in d_rows],
            )

        with self._path.open("r", encoding="utf-8") as f:
            ln_iter = ((idx, ln.strip()) for idx, ln in enumerate(f, start=1))

            while True:
                header_rows = []
                while len(header_rows) < self._parse_cfg.header_length:
                    try:
                        ln_idx, ln_txt = next(ln_iter)
                    except StopIteration:
                        if header_rows:  # If we have a partial header at EOF
                            raise ValueError(
                                f"[{self._file_label} HEADER {header_rows[0][0]}] Incomplete header at end of file."
                            )
                        return
                    if not ln_txt:
                        continue
                    if _is_footer(ln_txt):
                        continue
                    if not _is_header(ln_txt):
                        raise ValueError(
                            f"[{self._file_label} LINE {ln_idx}] Expected a header line, but got: '{ln_txt}'"
                        )
                    header_rows.append(
                        (
                            str(ln_idx),
                            ln_txt.lstrip(self._parse_cfg.header_prefix)
                            .strip()
                            .split(),
                        )
                    )

                data_rows = []
                for ln_idx, ln_txt in ln_iter:
                    if not ln_txt:
                        continue
                    if _is_header(ln_txt) or _is_footer(ln_txt):
                        yield _yield_block(header_rows, data_rows)
                        ln_iter = itertools.chain([(ln_idx, ln_txt)], ln_iter)
                        break
                    data_rows.append((ln_idx, ln_txt.split()))

                else:
                    if data_rows or header_rows:
                        yield _yield_block(header_rows, data_rows)
                    return

    @staticmethod
    def _apply_schema(
        tokens: List[str],
        schema: Dict[str, Tuple[Index, Converter]],
        is_strict: bool = True,
    ) -> Dict[str, Any]:
        result_dict = {}
        for key, (idx, converter) in schema.items():
            try:
                result_dict[key] = (
                    converter(tokens[idx])
                    if isinstance(idx, int)
                    else converter([tokens[i] for i in idx])
                )
            except (IndexError, ValueError) as e:
                exception_msg = f"Failed to parse key '{key}' with index '{idx}': {e}."
                if is_strict:
                    raise ValueError(exception_msg) from e
                else:
                    warnings.warn(
                        f"{exception_msg} Setting value to None.", UserWarning
                    )
                    result_dict[key] = None
        return result_dict
