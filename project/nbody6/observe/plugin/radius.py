from nbody6.observe.observer import PseudoObserverPluginBase
from nbody6.observe.snapshot import PseudoObservedSnapshot
from nbody6.utils.calc.cluster import calc_half_mass_radius


class HalfMassRadiusCalculator(PseudoObserverPluginBase):
    def __call__(self, snapshot: PseudoObservedSnapshot) -> PseudoObservedSnapshot:
        obs_df = snapshot.observation
        header = snapshot.header

        half_mass_radius = calc_half_mass_radius(
            stars_df=obs_df,
            center_coords_pc=header["density_center"],
        )
        if half_mass_radius <= 0:
            raise ValueError(
                f"[{snapshot.time} Myr] Calculated half-mass radius is non-positive: {half_mass_radius}"
            )

        # update header
        snapshot.header = {
            **header,
            "r_half_mass": float(half_mass_radius),
        }
        snapshot.observation["dist_dc_r_half_mass"] = (
            obs_df["dist_dc_pc"] / half_mass_radius
        )

        return snapshot

    def __repr__(self):
        return f"{type(self).__name__}(M(r)=\sum_{{i: r_i<=r}} m_i => solves M(r_h)= M_total / 2)"


class TwiceTidalRadiusCut(PseudoObserverPluginBase):
    def __call__(self, snapshot: PseudoObservedSnapshot) -> PseudoObservedSnapshot:
        obs_df = snapshot.observation

        # apply cut
        obs_df = obs_df[obs_df["dist_dc_r_tidal"] <= 2].copy()
        snapshot.observation = obs_df.reset_index(drop=True)

        return snapshot

    def __repr__(self):
        return f"{type(self).__name__}(cut: dist_dc_r_tidal <= 2)"
