from nbody6.assemble.assembler import SnapshotAssemblerPluginBase
from nbody6.assemble.snapshot import Snapshot
from nbody6.utils.calc import calc_log_surface_flux_ratio


class BasicPhotometryCalculator(SnapshotAssemblerPluginBase):
    def __call__(self, snapshot: Snapshot, **kwargs) -> Snapshot:
        star_df = snapshot.stars

        # calculate surface flux ratio to the Sun in log scale, given the log scale of T_eff in K
        star_df["log_F_F_sol"] = star_df["log_T_eff_K"].apply(
            lambda log_T: calc_log_surface_flux_ratio(log_T)
        )
        return snapshot

    def __repr__(self):
        return f"{type(self).__name__}(log10(F_star / F_sol) = 4 * (log10(T_star) - log10(T_sol)))"
