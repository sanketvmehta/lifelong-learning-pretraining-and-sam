import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import numpy as np
import matplotlib
from matplotlib import rc
from collections import defaultdict
from matplotlib.ticker import FormatStrFormatter
import matplotlib.patches as mpatches


def plot_contour(
    grid,
    values,
    coords,
    labels,
    path="./fig",
    cmap="magma_r",
    save=False,
    increment=0.3,
    num_levels=8,
    start=0,
):
    sns.set(style="ticks")
    sns.set_context(
        "paper",
        rc={
            "lines.linewidth": 2.5,
            "xtick.labelsize": 22,
            "ytick.labelsize": 22,
            "lines.markersize": 15,
            "legend.fontsize": 24,
            "axes.labelsize": 22,
            "axes.titlesize": 22,
            "legend.handlelength": 1,
            "legend.handleheight": 1,
        },
    )
    rc("text", usetex=False)
    matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
    plt.figure(figsize=(6.13, 4.98))
    formatter = FormatStrFormatter("%1.1f")

    levels = [start + l * increment for l in range(num_levels)]
    norm = matplotlib.colors.Normalize(start, increment * num_levels)
    contour = plt.contour(
        grid[:, :, 0], grid[:, :, 1], values, cmap=cmap, levels=levels, norm=norm
    )
    contourf = plt.contourf(
        grid[:, :, 0], grid[:, :, 1], values, cmap=cmap, levels=levels, norm=norm
    )
    colorbar = plt.colorbar(contourf, format="%.2g")
    for idx, coord in enumerate(coords):
        plt.scatter(coord[0], coord[1], marker="o", c="k", s=120, zorder=2)
        plt.text(coord[0] + 0.05, coord[1] + 0.05, labels[idx], fontsize=22)

    plt.margins(0.0)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.gca().xaxis.set_major_formatter(formatter)
    colorbar.ax.tick_params(labelsize=20)
    colorbar.ax.yaxis.set_major_formatter(formatter)

    if save:
        plt.savefig(path + ".png", dpi=200, bbox_inches="tight")
        plt.savefig(path + ".pdf", dpi=200, bbox_inches="tight")
    plt.show()
    plt.close()


def create_contour_plot(file, output_file, step=0.3, max_val=8, start=0, save=False):
    with open(file) as f:
        data = json.load(f)
    for key in data:
        data[key] = np.array(data[key])
    plot_contour(
        data["grid"],
        data["losses"],
        data["coords"],
        ["$w_1$", "$w_2$", "$w_3$"],
        path=output_file,
        save=save,
        increment=step,
        num_levels=int(-(-max_val // step)),  # Ceiling division
        start=start,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--analysis", required=True, choices=["lmi", "sharpness", "contour"]
    )
    parser.add_argument("--step", type=float)
    parser.add_argument("--max", type=float)
    args = parser.parse_args()

    if args.analysis == "contour":
        create_contour_plot(args.input, args.output, args.step, args.max, save=True)
    else:
        raise ValueError(f"Analysis type {args.analysis} not supported")


if __name__ == "__main__":
    main()