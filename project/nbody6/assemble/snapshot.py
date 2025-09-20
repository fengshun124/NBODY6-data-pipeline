from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Union, List

import pandas as pd


@dataclass(slots=True)
class Snapshot:
    time: float
    header: Dict[str, Union[int, float, str]]
    stars: pd.DataFrame
    binary_systems: pd.DataFrame

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"time={self.time}, "
            f"num_stars={len(self.stars)}, "
            f"num_binary_systems={len(self.binary_systems)}"
            ")"
        )

    def __eq__(self, value):
        if not isinstance(value, Snapshot):
            return NotImplemented
        return (
            self.time == value.time
            and self.header == value.header
            and self.stars.equals(value.stars)
            and self.binary_systems.equals(value.binary_systems)
        )

    def to_dict(self, is_materialize: bool = False) -> Dict:
        if is_materialize:
            return {
                "time": self.time,
                "header": self.header,
                "stars": self.stars.to_dict(orient="records"),
                "binary_systems": self.binary_systems.to_dict(orient="records"),
            }
        else:
            return {
                "time": self.time,
                "header": self.header,
                "stars": self.stars,
                "binary_systems": self.binary_systems,
            }

    @classmethod
    def from_dict(cls, data: Dict) -> "Snapshot":
        return cls(
            time=data["time"],
            header=data["header"],
            stars=pd.DataFrame(data["stars"]),
            binary_systems=pd.DataFrame(data["binary_systems"]),
        )

    @classmethod
    def from_pickle(cls, filepath: Union[str, Path]) -> "Snapshot":
        filepath = Path(filepath)
        with open(filepath, "rb") as f:
            data = pd.read_pickle(f)
        return cls.from_dict(data)


@dataclass(slots=True)
class SnapshotSeries:
    root: Path
    snapshots: Dict[float, Snapshot]

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"root={self.root}, "
            f"num_snapshots={len(self.snapshots)}"
            ")"
        )

    def __len__(self) -> int:
        return len(self.snapshots)

    def __getitem__(self, time: float) -> Snapshot:
        return self.snapshots[time]

    def __iter__(self):
        return iter(self.snapshots.items())

    def __eq__(self, value):
        if not isinstance(value, SnapshotSeries):
            return NotImplemented
        return self.root == value.root and self.snapshots == value.snapshots

    @property
    def timestamps(self) -> List[float]:
        return list(self.snapshots.keys())

    def to_dict(self, is_materialize: bool = False) -> Dict:
        if is_materialize:
            return {
                "root": str(self.root),
                "snapshots": {
                    time: snapshot.to_dict(is_materialize=True)
                    for time, snapshot in self.snapshots.items()
                },
            }
        else:
            return {
                "root": str(self.root),
                "snapshots": self.snapshots,
            }

    def to_pickle(self, filepath: Union[str, Path], is_materialize: bool = False):
        filepath = Path(filepath)
        data = self.to_dict(is_materialize=is_materialize)
        with open(filepath, "wb") as f:
            pd.to_pickle(data, f)

    @classmethod
    def from_dict(cls, data: Dict) -> "SnapshotSeries":
        return cls(
            root=Path(data["root"]),
            snapshots={
                float(time): Snapshot.from_dict(snapshot_data)
                for time, snapshot_data in data["snapshots"].items()
            },
        )

    @classmethod
    def from_pickle(cls, filepath: Union[str, Path]) -> "SnapshotSeries":
        filepath = Path(filepath)
        with open(filepath, "rb") as f:
            data = pd.read_pickle(f)
        return cls.from_dict(data)
