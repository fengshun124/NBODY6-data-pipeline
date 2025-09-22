import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Union

import pandas as pd

from nbody6.utils.calc.cluster import Coord3D


@dataclass(slots=True)
class PseudoObservedSnapshot:
    time: float
    simulated_galactic_center: Coord3D
    observation: pd.DataFrame
    header: Dict[str, Union[int, float, str]]
    stars: pd.DataFrame
    binary_systems: pd.DataFrame

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"time={self.time}, "
            f"num_observed_stars={len(self.observation)}, "
            f"num_stars={len(self.stars)}, "
            f"num_binary_systems={len(self.binary_systems)}"
            ")"
        )

    def __eq__(self, value):
        if not isinstance(value, PseudoObservedSnapshot):
            return NotImplemented
        return (
            self.time == value.time
            and self.simulated_galactic_center == value.simulated_galactic_center
            and self.observation.equals(value.observation)
            and self.header == value.header
            and self.stars.equals(value.stars)
            and self.binary_systems.equals(value.binary_systems)
        )

    def to_dict(self, is_materialize: bool = True) -> dict:
        if is_materialize:
            return {
                "time": float(self.time),
                "simulated_galactic_center": list(self.simulated_galactic_center),
                "observation": self.observation.to_dict(orient="records"),
                "header": dict(self.header),
                "stars": self.stars.to_dict(orient="records"),
                "binary_systems": self.binary_systems.to_dict(orient="records"),
            }
        else:
            return {
                "time": self.time,
                "simulated_galactic_center": self.simulated_galactic_center,
                "observation": self.observation,
                "header": self.header,
                "stars": self.stars,
                "binary_systems": self.binary_systems,
            }

    @classmethod
    def from_dict(cls, data: dict) -> "PseudoObservedSnapshot":
        galactic_center = data["simulated_galactic_center"]

        return cls(
            time=float(data["time"]),
            simulated_galactic_center=tuple(galactic_center)
            if isinstance(galactic_center, (list, tuple))
            else galactic_center,
            observation=pd.DataFrame(data["observation"])
            if isinstance(data["observation"], list)
            else data["observation"],
            header=dict(data["header"]),
            stars=pd.DataFrame(data["stars"])
            if isinstance(data["stars"], list)
            else data["stars"],
            binary_systems=pd.DataFrame(data["binary_systems"])
            if isinstance(data["binary_systems"], list)
            else data["binary_systems"],
        )

    @classmethod
    def from_pickle(cls, filepath: Union[str, Path]) -> "PseudoObservedSnapshot":
        filepath = Path(filepath)
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        return cls.from_dict(data)


@dataclass(slots=True)
class PseudoObservedSnapshotSeries:
    root: Path
    snapshots: Dict[float, PseudoObservedSnapshot]

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"root={self.root}, "
            f"num_snapshots={len(self.snapshots)}"
            ")"
        )

    def __len__(self) -> int:
        return len(self.snapshots)

    def __getitem__(self, time: float) -> PseudoObservedSnapshot:
        return self.snapshots[time]

    def __iter__(self):
        return iter(self.snapshots.items())

    def __eq__(self, value):
        if not isinstance(value, PseudoObservedSnapshotSeries):
            return NotImplemented
        return self.root == value.root and self.snapshots == value.snapshots

    @property
    def timestamps(self) -> list[float]:
        return sorted(self.snapshots.keys())

    def to_dict(self, is_materialize: bool = True) -> dict:
        """
        Convert pseudo-observed snapshot series to dictionary.

        Args:
            is_materialize: If True, converts all data to serializable format.
        """
        if is_materialize:
            return {
                "root": str(self.root),
                "snapshots": {
                    str(time): snapshot.to_dict(is_materialize=True)
                    for time, snapshot in self.snapshots.items()
                },
            }
        else:
            return {
                "root": str(self.root),
                "snapshots": self.snapshots,
            }

    def to_pickle(self, filepath: Union[str, Path], is_materialize: bool = True):
        filepath = Path(filepath)
        data = self.to_dict(is_materialize=is_materialize)
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def from_dict(cls, data: dict) -> "PseudoObservedSnapshotSeries":
        return cls(
            root=Path(data["root"]),
            snapshots={
                float(time): PseudoObservedSnapshot.from_dict(snapshot_data)
                for time, snapshot_data in data["snapshots"].items()
            },
        )

    @classmethod
    def from_pickle(cls, filepath: Union[str, Path]) -> "PseudoObservedSnapshotSeries":
        filepath = Path(filepath)
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        return cls.from_dict(data)
