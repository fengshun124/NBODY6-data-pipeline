import numpy as np
from astropy.constants import L_bol0, L_sun


def calc_log_surface_flux_ratio(log_T_eff_K, log_T_sol_K: float = np.log10(5772)):
    # F = \sigma_SB * T^4
    # => F_star / F_sol = (T_star / T_sol)^4
    # => log10(F_star / F_sol) = 4 * (log10(T_star) - log10(T_sol))
    return float(4 * (np.asarray(log_T_eff_K) - log_T_sol_K))


def calc_effective_temperature_K(L_L_sol, R_R_sol, T_sol_K: float = 5772):
    # L = 4 * pi * R^2 * \sigma_SB * T^4
    # => T_eff = (L / (4 * pi * R^2 * \sigma_SB))^(1/4)
    # => T_eff = (L_ratio / R_ratio^2)^(1/4) * T_sol
    return (np.asarray(L_L_sol) / np.asarray(R_R_sol) ** 2) ** (1 / 4) * T_sol_K


def calc_log_effective_temperature_K(
    log_L_L_sol, log_R_R_sol, log_T_sol_K: float = np.log10(5772)
):
    # L = 4 * pi * R^2 * \sigma_SB * T^4
    # => T_eff = (L / (4 * pi * R^2 * \sigma_SB))^(1/4)
    # => log10(T_eff) = (1/4) * (log10(L) - 2*log10(R)) + log10(T_sol)
    return np.log10(
        calc_effective_temperature_K(10**log_L_L_sol, 10**log_R_R_sol, 10**log_T_sol_K)
    )


def calc_bolometric_magnitude(log_L_L_sol):
    # M_bol - M_bol_sun = -2.5 * log10(L / L_sun)
    # => M_bol = M_bol_sun - 2.5 * log10(L / L_sun)
    M_bol_sun = -2.5 * np.log10(L_sun / L_bol0)

    return -2.5 * log_L_L_sol + M_bol_sun


def calc_apparent_magnitude(abs_mag, dist_pc):
    # m - M = 5 * log10(d / 10 pc)
    # => m = M + 5 * log10(d / 10 pc)
    return abs_mag + 5 * (np.log10(dist_pc) - 1)
