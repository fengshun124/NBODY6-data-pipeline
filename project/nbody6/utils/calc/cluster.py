from typing import List, Tuple, Union

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import (
    SkyCoord,
    CartesianRepresentation,
    CartesianDifferential,
    SkyOffsetFrame,
)

Coord3D = Union[List[float], Tuple[float, float, float], np.ndarray]


def calc_half_mass_radius(
    stars_df: pd.DataFrame,
    center_coords_pc: Coord3D,
    mass_key: str = "mass",
    x_pc_key: str = "x",
    y_pc_key: str = "y",
    z_pc_key: str = "z",
) -> float:
    distances = np.linalg.norm(
        stars_df[[x_pc_key, y_pc_key, z_pc_key]].values
        - np.asarray(center_coords_pc)[np.newaxis, :],
        axis=1,
    )
    sorted_idx = np.argsort(distances)
    cum_mass = np.cumsum(stars_df[mass_key].values[sorted_idx])
    half_mass = cum_mass[-1] / 2
    half_idx = np.searchsorted(cum_mass, half_mass)
    return distances[sorted_idx][half_idx]


def convert_to_offset_frame(
    cluster_center_pc: Coord3D,
    centered_stars_df: pd.DataFrame,
    x_pc_key: str = "x",
    y_pc_key: str = "y",
    z_pc_key: str = "z",
    vx_kms_key: str = "vx",
    vy_kms_key: str = "vy",
    vz_kms_key: str = "vz",
) -> pd.DataFrame:
    c = np.asarray(cluster_center_pc)

    world_coords = SkyCoord(
        CartesianRepresentation(
            x=(c[0] + centered_stars_df[x_pc_key].values) * u.pc,
            y=(c[1] + centered_stars_df[y_pc_key].values) * u.pc,
            z=(c[2] + centered_stars_df[z_pc_key].values) * u.pc,
        ).with_differentials(
            CartesianDifferential(
                d_x=centered_stars_df[vx_kms_key].values * u.km / u.s,
                d_y=centered_stars_df[vy_kms_key].values * u.km / u.s,
                d_z=centered_stars_df[vz_kms_key].values * u.km / u.s,
            )
        ),
        frame="galactic",
    )

    origin_coord = SkyCoord(
        CartesianRepresentation(
            x=c[0] * u.pc,
            y=c[1] * u.pc,
            z=c[2] * u.pc,
        ),
        frame="galactic",
    )

    offset_coords = world_coords.transform_to(SkyOffsetFrame(origin=origin_coord))

    return centered_stars_df.copy().assign(
        lon_deg=offset_coords.lon.deg,
        lat_deg=offset_coords.lat.deg,
        pm_lon_coslat_mas_yr=offset_coords.pm_lon_coslat.to(u.mas / u.yr).value,
        pm_lat_mas_yr=offset_coords.pm_lat.to(u.mas / u.yr).value,
        dist_pc=offset_coords.distance.pc,
        rv_kms=offset_coords.radial_velocity.to(u.km / u.s).value,
    )
