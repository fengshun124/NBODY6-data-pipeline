import warnings
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from nbody6.calc.summary import summarize_timestamp_stats
from nbody6.parser import (
    DensityCenterParser,
    FileParserBase,
    Fort19Parser,
    Fort82Parser,
    Fort83Parser,
    OUT9Parser,
    OUT34Parser,
)


@dataclass(slots=True)
class NBody6Data:
    root: Path
    # file parsers
    parser_dict: dict[str, FileParserBase]
    # timestamp info
    timestamps: list[float]
    raw_timestamp_df: pd.DataFrame

    def __repr__(self):
        parsers_str = (
            "{\n"
            + ",\n".join(
                f"        {k!r}: {repr(v).replace(str(self.root), '...')}"
                for k, v in self.parser_dict.items()
            )
            + "\n    }"
        )
        return (
            f"{type(self).__name__}(\n"
            f"    root={self.root!r},\n"
            f"    timestamp_stats={self.timestamp_stats},\n"
            f"    parsers={parsers_str}\n"
            f")"
        )

    @property
    def timestamp_stats(self) -> dict[str, dict[str, float | None]]:
        return summarize_timestamp_stats(self.timestamps)

    def __getitem__(self, timestamp: float) -> dict[str, FileParserBase]:
        return {name: parser[timestamp] for name, parser in self.parser_dict.items()}


class NBODY6DataLoader:
    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        if not self._root.is_dir():
            raise NotADirectoryError(f"Root path '{self._root}' is not a directory.")

        self._parser_dict: dict[str, FileParserBase] = {
            "fort.19": Fort19Parser(self._root / "fort.19"),
            "fort.82": Fort82Parser(self._root / "fort.82"),
            "fort.83": Fort83Parser(self._root / "fort.83"),
            "OUT34": OUT34Parser(self._root / "OUT34"),
            "OUT9": OUT9Parser(self._root / "OUT9"),
            "densCentre.txt": DensityCenterParser(self._root / "densCentre.txt"),
        }
        self._validate_file()

        self._simulation_data: NBody6Data | None = None

    def __repr__(self):
        return f"{type(self).__name__}(root={self._root})"

    def _validate_file(self) -> None:
        for name, parser in self._parser_dict.items():
            if not parser.path.is_file():
                raise FileNotFoundError(
                    f"Required file '{parser.path.name}' not found in '{self._root}'."
                )

    @property
    def parser_dict(self) -> dict[str, FileParserBase]:
        return self._parser_dict

    @property
    def simulation_data(self) -> NBody6Data | None:
        if self._simulation_data is None:
            warnings.warn(
                f"[{self._root.name}] Simulation not loaded. Call 'load()' to parse data.",
                UserWarning,
            )
        return self._simulation_data

    def load(
        self,
        is_strict: bool = True,
        is_verbose: bool = True,
        is_allow_timestamp_trim: bool = False,
        timestamp_tolerance: float = 2e-2,
    ) -> NBody6Data:
        if self._simulation_data is not None:
            warnings.warn(
                f"[{self._root}] Reloading simulation data. Previous data will be overwritten.",
                UserWarning,
            )
            self._simulation_data = None

        if not is_strict:
            warnings.warn(
                f"[{self._root}] Non-strict mode enabled. Schema errors will be ignored.",
                UserWarning,
            )
        if is_allow_timestamp_trim:
            warnings.warn(
                f"[{self._root}] Timestamp trimming enabled. Unmatched timestamps will be discarded.",
                UserWarning,
            )
        if timestamp_tolerance < 0:
            raise ValueError("Timestamp tolerance must be non-negative.")

        for parser in (
            pbar := tqdm(
                self._parser_dict.values(),
                disable=not is_verbose,
                dynamic_ncols=True,
                leave=False,
            )
        ):
            pbar.set_description(f"Loading {parser._path.name}")
            try:
                parser.parse(is_strict=is_strict)
            except Exception as e:
                raise RuntimeError(
                    f"[{parser.path.name}] Error parsing file: {e}"
                ) from e

        timestamp_df = pd.DataFrame.from_dict(
            {name: p.timestamps for name, p in self._parser_dict.items()},
            orient="index",
        ).T

        if not is_allow_timestamp_trim:
            # strict: timestamps across ALL files must match
            ts_counts = timestamp_df.count()
            if ts_counts.nunique() > 1:
                raise ValueError(
                    f"[{self._root.name}] Timestamps count mismatch across files. "
                    f"Counts: {ts_counts.to_dict()}"
                )

            ts_diffs = timestamp_df.max(axis=1) - timestamp_df.min(axis=1)
            inconsistent_rows = ts_diffs[ts_diffs > timestamp_tolerance]
            if not inconsistent_rows.empty:
                affected_indices = inconsistent_rows.index.tolist()
                display_indices = affected_indices[:5] + (
                    ["..."] if len(affected_indices) > 5 else []
                )
                raise ValueError(
                    f"[{self._root.name}] {len(inconsistent_rows)} inconsistent timestamps found "
                    f"(tolerance: {timestamp_tolerance}). Mismatched row indices: {display_indices}. "
                    f"Max difference is {inconsistent_rows.max():.2e}."
                )

            # unify timestamps using OUT34 as reference
            ref_ts_indices = timestamp_df.index.to_list()
            ref_timestamps = timestamp_df.loc[ref_ts_indices, "OUT34"].round(2).tolist()

        else:
            # allow trimming: keep timestamps that appear in ALL files, MUST starting from 0
            trimmed_timestamp_df = timestamp_df.dropna().loc[
                lambda df: (df.max(axis=1) - df.min(axis=1)) <= timestamp_tolerance
            ]
            if trimmed_timestamp_df.empty:
                timestamp_stats = {
                    name: summarize_timestamp_stats(parser.timestamps)
                    for name, parser in self._parser_dict.items()
                }
                raise ValueError(
                    f"[{self._root.name}] No aligned timestamps found across all files "
                    f"with tolerance {timestamp_tolerance}. "
                    f"Timestamp stats: {timestamp_stats}"
                )
            # unify timestamps using OUT34 as reference
            ref_ts_indices = trimmed_timestamp_df.index.to_list()
            ref_timestamps = (
                trimmed_timestamp_df.loc[ref_ts_indices, "OUT34"].round(2).tolist()
            )

        if ref_timestamps[0] != 0:
            warnings.warn(
                f"[{self._root.name}] First aligned timestamp is {ref_timestamps[0]}, not 0.0."
            )

        # unify timestamps across all parsers
        aligned_timestamp_df = timestamp_df.iloc[ref_ts_indices]
        for name, parser in self._parser_dict.items():
            for ref_ts, src_ts in zip(ref_timestamps, aligned_timestamp_df[name]):
                if src_ts != ref_ts:
                    parser.update_timestamp(float(src_ts), float(ref_ts))

        self._simulation_data = NBody6Data(
            root=self._root,
            parser_dict=self._parser_dict,
            timestamps=ref_timestamps,
            raw_timestamp_df=timestamp_df,
        )
        return self._simulation_data
