from typing import List, Union

import numpy as np
from astropy import units as u
from astropy.constants import G, M_sun

LN10 = np.log(10.0)


def calc_total_log_luminosity(log_L_L_sol1: float, log_L_L_sol2: float) -> float:
    # L_total = L1 + L2
    # L_total/L_sol = 10**log_L_L_sol1 + 10**log_L_L_sol2
    # ---
    # Use log-sum-exp in ln-space for numerical stability.
    # log10(L_total/L_sol) = log10(10**log_L_L_sol1 + 10**log_L_L_sol2) [log10(L_sol)]
    # = ln(10**log_L_L_sol1 + 10**log_L_L_sol2) / ln(10)
    return float(np.logaddexp(log_L_L_sol1 * LN10, log_L_L_sol2 * LN10) / LN10)


def calc_equivalent_radius(r_R_sol1: float, r_R_sol2: float) -> float:
    # r_eq^2 = r1^2 + r2^2 [R_sol^2]
    # r_eq = sqrt(r1^2 + r2^2) [R_sol]
    return float(np.sqrt(r_R_sol1**2 + r_R_sol2**2))


def calc_log_equivalent_radius(log_R_R_sol1: float, log_R_R_sol2: float) -> float:
    # r_eq = sqrt((10**log_r1)^2 + (10**log_r2)^2) [R_sol]
    # log_r_eq = log10(r_eq) [log10(R_sol)]
    # ---
    # Use log-sum-exp in ln-space for numerical stability.
    # log10(sqrt(10^(2*log_r1) + 10^(2*log_r2))) = 0.5 * log10(10^(2*log_r1) + 10^(2*log_r2))
    # = 0.5 * (ln(10^(2*log_r1) + 10^(2*log_r2)) / ln(10))
    return float(
        0.5 * (np.logaddexp(2 * log_R_R_sol1 * LN10, 2 * log_R_R_sol2 * LN10) / LN10)
    )


def calc_total_mass(mass_M_sol1: float, mass_M_sol2: float) -> float:
    # m_total = m1 + m2 [M_sol]
    return mass_M_sol1 + mass_M_sol2


def calc_photocentric(
    L_L_sol1: float,
    L_L_sol2: float,
    vec1: Union[List[float], np.ndarray],
    vec2: Union[List[float], np.ndarray],
) -> np.ndarray:
    # calculate the luminosity-weighted average vector
    L_total = L_L_sol1 + L_L_sol2
    if np.isclose(L_total, 0):
        # If total luminosity is zero, return the geometric center
        return (np.asarray(vec1) + np.asarray(vec2)) / 2
    return (L_L_sol1 * np.asarray(vec1) + L_L_sol2 * np.asarray(vec2)) / L_total


def calc_semi_major_axis(
    mass_M_sol1: float, mass_M_sol2: float, period_days: float
) -> float:
    # Kepler's Third Law: a^3 = (G * (m1 + m2) * P^2) / (4 * pi^2) [m^3]
    # a = ((G * (m1 + m2) * P^2) / (4 * pi^2))^(1/3) [m]
    total_mass = (mass_M_sol1 + mass_M_sol2) * M_sun
    period = period_days * u.day

    a = ((G * total_mass * period**2) / (4 * np.pi**2)) ** (1 / 3)
    return float(a.to(u.AU).value)


def calc_orbital_plane_inclination_rad(
    pos1_pc: Union[List[float], np.ndarray],
    pos2_pc: Union[List[float], np.ndarray],
    vel1_kms: Union[List[float], np.ndarray],
    vel2_kms: Union[List[float], np.ndarray],
) -> float:
    # Calculate relative position and velocity vectors
    # r = r2 - r1 [pc]
    r_relative = np.asarray(pos2_pc) - np.asarray(pos1_pc)
    # v = v2 - v1 [km/s]
    v_relative = np.asarray(vel2_kms) - np.asarray(vel1_kms)

    # Calculate the specific angular momentum vector: h = r x v [pc * km/s]
    h = np.cross(r_relative, v_relative)
    h_norm = np.linalg.norm(h)

    # If h_norm is zero, the orbit is radial (r and v are parallel).
    # the inclination is ill-defined; return 0
    if np.isclose(h_norm, 0):
        return 0.0

    # inclination 'i' is the angle between the angular momentum vector 'h' and the z-axis.
    # i = arccos(h_z / |h|) [rad]
    return float(np.arccos(np.clip(h[2] / h_norm, -1.0, 1.0)))


def is_wide_binary(semi_major_axis_au: float, threshold_au: float = 1000.0) -> bool:
    return semi_major_axis_au > threshold_au


def is_hard_binary(
    semi_major_axis_au: float,
    half_mass_radius_pc: float,
    num_stars: int,
):
    # Hard binary condition: a < r_hm / N (Heggie's law) [AU]
    # convert pc to AU for comparison
    a_threshold_au = half_mass_radius_pc * u.pc.to(u.AU) / num_stars
    return semi_major_axis_au < a_threshold_au
