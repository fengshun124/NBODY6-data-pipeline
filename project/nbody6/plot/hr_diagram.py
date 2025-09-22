from typing import Callable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.legend_handler import HandlerTuple

from nbody6.plot import PLOT_STYLE_DICT


def default_mass2size(mass):
    return 12 + 64 * np.log1p(mass)


def plot_hr_diagram(
    data_df: pd.DataFrame,
    hr_ax: Optional[plt.Axes] = None,
    log_T_eff_key: str = "log_T_eff_K",
    log_L_L_sol_key: str = "log_L_L_sol",
    mass_key: str = "mass",
    mass_mapper: Callable = default_mass2size,
    is_add_type_legend: bool = True,
    is_add_mass_legend: bool = True,
    mass_ref_list: Optional[List[float]] = None,
    header: Optional[str] = None,
):
    if not hr_ax:
        fig, hr_ax = plt.subplots(figsize=(5, 6), dpi=300)

    data_dict = {
        "resolved": data_df[~data_df["is_unresolved_binary"] & data_df["is_binary"]],
        "single": data_df[~data_df["is_binary"]],
        "unresolved": data_df[data_df["is_unresolved_binary"]],
    }

    for key, sub_df in data_dict.items():
        if not sub_df.empty:
            hr_ax.scatter(
                sub_df[log_T_eff_key],
                sub_df[log_L_L_sol_key],
                s=mass_mapper(sub_df[mass_key]),
                label=key,
                edgecolors=PLOT_STYLE_DICT[key]["color"],
                facecolors="none",
                alpha=0.6,
                marker=PLOT_STYLE_DICT[key]["marker"],
                lw=1,
            )

    hr_ax.set_xlabel(r"$\log(T_{\mathrm{eff}}/[\mathrm{K}])$")
    hr_ax.set_xlim(5.3, 3.1)
    hr_ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
    hr_ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))

    hr_ax.set_ylabel(r"$\log(L/L_{\odot})$")
    hr_ax.set_ylim(-5, 7)
    hr_ax.yaxis.set_major_locator(plt.MultipleLocator(2))
    hr_ax.yaxis.set_minor_locator(plt.MultipleLocator(1))

    hr_ax.grid(visible=True, linestyle=":", linewidth=0.5, alpha=0.4, c="k")

    # legend
    all_handles, all_labels = [], []
    if is_add_type_legend:
        # phantom entry for header
        all_handles.append(plt.Line2D([], [], color="none", marker=""))
        all_labels.append("\n$\\bf{Type}$")

        type_handles, type_labels =hr_ax.get_legend_handles_labels()
        all_handles.extend(type_handles)
        all_labels.extend(type_labels)

    if is_add_mass_legend:
        mass_ref_list = mass_ref_list or [0.05, 0.1, 0.5, 1, 2, 4, 8]

        all_handles.append(plt.Line2D([], [], color="none", label="\n$\\bf{Mass}$"))
        all_labels.append("\n$\\bf{Mass}$")
        mass_handles, mass_labels = zip(
            *[
                (
                    tuple(
                       hr_ax.scatter(
                            [],
                            [],
                            marker=PLOT_STYLE_DICT[key]["marker"],
                            s=mass_mapper(mass),
                            c="None",
                            edgecolors=PLOT_STYLE_DICT[key]["color"],
                            linewidths=0.8,
                        )
                        for key in PLOT_STYLE_DICT
                    ),
                    f"{mass:.2f} $M_\\odot$",
                )
                for mass in mass_ref_list
            ]
        )
        all_handles.extend(mass_handles)
        all_labels.extend(mass_labels)

    if all_handles and all_labels:
        legend = hr_ax.legend(
            all_handles,
            all_labels,
            handler_map={tuple: HandlerTuple(ndivide=None, pad=1)},
            handletextpad=0.8,
            handlelength=4 if is_add_mass_legend else 1,
            loc="center left",
            frameon=True,
            scatterpoints=True,
            labelspacing=1,
            borderpad=1,
            fontsize=12,
            bbox_to_anchor=(1.04, 0.5),
        )
        for item, label in zip(legend.legend_handles, legend.texts):
            if label.get_text() in ["\n$\\bf{Type}$", "\n$\\bf{Mass}$"]:
                width = item.get_window_extent(hr_ax.figure.canvas.get_renderer()).width
                label.set_ha("left")
                label.set_position((-2 * width, 0))
                item.set_visible(False)
                label.set_fontweight("bold")
                label.set_fontsize(14)
                label.set_color("black")

    if header:
       hr_ax.set_title(header, fontsize=16, pad=8)

    return hr_ax
