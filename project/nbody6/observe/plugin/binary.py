from nbody6.observe.observer import PseudoObserverPluginBase
from nbody6.observe.snapshot import PseudoObservedSnapshot
from nbody6.utils.calc.binary import is_hard_binary, is_wide_binary


class BinaryClassifier(PseudoObserverPluginBase):
    def __call__(self, snapshot: PseudoObservedSnapshot) -> PseudoObservedSnapshot:
        bin_sys_df = snapshot.binary_systems.copy()

        semi_major_axis_au = bin_sys_df["semi"].to_numpy()
        half_mass_radius_pc = snapshot.header["r_half_mass"]
        num_stars = len(snapshot.observation)

        bin_sys_df["is_hard_binary"] = is_hard_binary(
            semi_major_axis_au=semi_major_axis_au,
            half_mass_radius_pc=half_mass_radius_pc,
            num_stars=num_stars,
        )
        bin_sys_df["is_wide_binary"] = is_wide_binary(
            semi_major_axis_au=semi_major_axis_au
        )
        snapshot.binary_systems = bin_sys_df
        return snapshot

    def __repr__(self):
        return f"{type(self).__name__}(is_hard_binary: a < r_half_mass / N, is_wide_binary: a > 1000 AU)"
