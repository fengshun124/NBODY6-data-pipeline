from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd


def summarize_timestamp_stats(
    timestamps: Union[List[float], pd.Series],
) -> Dict[str, Union[int, float, None]]:
    if not timestamps:
        return {"count": 0, "min": None, "max": None, "step": None}
    return {
        "count": len(timestamps),
        "min": min(timestamps),
        "max": max(timestamps),
        "step": round(float(np.nanmean(np.diff(timestamps))), 2),
    }


def summarize_descriptive_stats(
    values: Union[List[Union[float, int]], pd.Series],
    key: str,
    prefix: Optional[str] = None,
) -> Dict[str, float]:
    prefix = f"{prefix}_" if prefix else ""
    s = pd.Series(values)

    return {
        f"{prefix}{key}_mean": s.mean(),
        f"{prefix}{key}_std": s.std(),
        f"{prefix}{key}_min": s.min(),
        f"{prefix}{key}_q1": s.quantile(0.25),
        f"{prefix}{key}_median": s.median(),
        f"{prefix}{key}_q3": s.quantile(0.75),
        f"{prefix}{key}_max": s.max(),
    }
