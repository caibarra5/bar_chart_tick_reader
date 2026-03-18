from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_mean_heatmap(
    qs_bar1_over_time,
    qs_bar2_over_time,
    env,
    outpath: Path,
    title="q(mean_height) over time",
):
    """
    Compute distribution over M = 0.5*(bar1 + bar2)
    and save heatmap.
    """

    T = len(qs_bar1_over_time)

    values = []
    for fine_idx in range(env.Nfine):
        tick = fine_idx // env.n_fine
        fine_within = fine_idx % env.n_fine

        lo = env.tick_values[tick]
        hi = env.tick_values[tick + 1]
        width = (hi - lo) / env.n_fine
        center_val = lo + (fine_within + 0.5) * width
        values.append(center_val)

    values = np.array(values)
    mean_dists = []

    for t in range(T):
        p1 = qs_bar1_over_time[t]
        p2 = qs_bar2_over_time[t]
        p_sum = np.convolve(p1, p2)

        sum_values = np.linspace(
            values[0] + values[0],
            values[-1] + values[-1],
            len(p_sum),
        )
        mean_values = 0.5 * sum_values
        mean_dists.append(p_sum)

    M = np.stack(mean_dists, axis=1)

    outpath.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.imshow(
        M,
        aspect="auto",
        origin="lower",
        extent=[0, T, mean_values[0], mean_values[-1]],
        cmap="gray_r",
    )
    plt.colorbar(label="probability")
    plt.xlabel("time")
    plt.ylabel("mean height (data units)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def save_probvec_png(p, outpath, title):
    p = np.asarray(p, dtype=float).ravel()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.bar(np.arange(len(p)), p)
    plt.title(title)
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def save_heatmap_time_state(
    qs_over_time,
    env,
    outpath: Path,
    title: str,
    xlabel: str = "time",
    ylabel: str = "bar height (data units)",
):
    """
    qs_over_time: list of (Nfine,) vectors
    env: BarChartEnv instance (needed for value mapping)
    """

    M = np.stack(qs_over_time, axis=1)

    values = []
    for fine_idx in range(env.Nfine):
        tick = fine_idx // env.n_fine
        fine_within = fine_idx % env.n_fine

        lo = env.tick_values[tick]
        hi = env.tick_values[tick + 1]

        width = (hi - lo) / env.n_fine
        center_val = lo + (fine_within + 0.5) * width
        values.append(center_val)

    values = np.array(values)

    outpath.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.imshow(
        M,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[0, M.shape[1], values[0], values[-1]],
        cmap="gray_r",
    )
    plt.colorbar(label="probability")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def save_heatmap_time_coarse(
    qs_over_time,
    env,
    outpath: Path,
    title: str,
    xlabel: str = "time",
    ylabel: str = "query value (data units)",
):
    """
    qs_over_time: list of (Ncoarse,) vectors
    env: BarChartEnv instance (needed for coarse-bin value mapping)
    """

    M = np.stack(qs_over_time, axis=1)

    values = []
    for coarse_idx in range(env.Ncoarse):
        tick = coarse_idx // env.n_coarse
        coarse_within = coarse_idx % env.n_coarse

        lo = env.tick_values[tick]
        hi = env.tick_values[tick + 1]

        width = (hi - lo) / env.n_coarse
        center_val = lo + (coarse_within + 0.5) * width
        values.append(center_val)

    values = np.array(values)

    outpath.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.imshow(
        M,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[0, M.shape[1], values[0], values[-1]],
        cmap="gray_r",
    )
    plt.colorbar(label="probability")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def save_categorical_heatmap(qs_over_time, outpath, title):
    M = np.stack(qs_over_time, axis=1)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.imshow(M, aspect="auto", origin="lower", cmap="gray_r")
    plt.colorbar(label="probability")
    plt.xlabel("time")
    plt.ylabel("state index")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
