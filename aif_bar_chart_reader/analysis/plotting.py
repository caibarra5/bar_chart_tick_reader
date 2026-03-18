from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _uniform_numeric_ticks(y_min, y_max, max_ticks=6):
    if np.isclose(y_min, y_max):
        return np.array([int(round(y_min))], dtype=int)

    span = y_max - y_min
    target_intervals = max(max_ticks - 1, 4)
    rough_step = max(span / target_intervals, 1.0)

    step_candidates = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    step = step_candidates[-1]
    for candidate in step_candidates:
        if candidate >= rough_step:
            step = candidate
            break

    tick_start = int(np.floor(y_min))
    tick_end = int(np.ceil(y_max))

    ticks = np.arange(tick_start, tick_end + 1, step, dtype=int)

    if ticks.size < 5:
        ticks = np.linspace(tick_start, tick_end, num=5)
        ticks = np.unique(np.round(ticks).astype(int))

    if ticks.size < 5:
        ticks = np.arange(tick_start, tick_end + 1, 1, dtype=int)

    if ticks.size == 0:
        ticks = np.array([tick_start, tick_end], dtype=int)

    return ticks


def _save_numeric_heatmap(
    M,
    outpath: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    y_min: float,
    y_max: float,
    cmap: str = "gray_r",
):
    outpath.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.imshow(
        M,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[0, M.shape[1], y_min, y_max],
        cmap=cmap,
    )
    plt.colorbar(label="probability")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    ticks = _uniform_numeric_ticks(y_min, y_max)
    plt.yticks(ticks)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def _fine_bin_centers(env):
    values = []
    for fine_idx in range(env.Nfine):
        tick = fine_idx // env.n_fine
        fine_within = fine_idx % env.n_fine

        lo = env.tick_values[tick]
        hi = env.tick_values[tick + 1]
        width = (hi - lo) / env.n_fine
        center_val = lo + (fine_within + 0.5) * width
        values.append(center_val)
    return np.array(values, dtype=float)


def _coarse_bin_centers(env):
    values = []
    for coarse_idx in range(env.Ncoarse):
        tick = coarse_idx // env.n_coarse
        coarse_within = coarse_idx % env.n_coarse

        lo = env.tick_values[tick]
        hi = env.tick_values[tick + 1]
        width = (hi - lo) / env.n_coarse
        center_val = lo + (coarse_within + 0.5) * width
        values.append(center_val)
    return np.array(values, dtype=float)


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

    values = _fine_bin_centers(env)
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
    _save_numeric_heatmap(
        M,
        outpath=outpath,
        title=title,
        xlabel="time",
        ylabel="mean height (data units)",
        y_min=float(mean_values[0]),
        y_max=float(mean_values[-1]),
    )


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

    values = _fine_bin_centers(env)
    _save_numeric_heatmap(
        M,
        outpath=outpath,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        y_min=float(values[0]),
        y_max=float(values[-1]),
    )


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

    values = _coarse_bin_centers(env)
    _save_numeric_heatmap(
        M,
        outpath=outpath,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        y_min=float(values[0]),
        y_max=float(values[-1]),
    )


def save_report_choice_heatmap(
    qs_over_time,
    env,
    outpath: Path,
    title: str,
    xlabel: str = "time",
    ylabel: str = "report value (data units)",
):
    """
    qs_over_time: list of (Ncoarse + 1,) vectors
    env: BarChartEnv instance (used to map report bins to data values)
    """

    M = np.stack(qs_over_time, axis=1)

    coarse_values = _coarse_bin_centers(env)
    coarse_width = coarse_values[1] - coarse_values[0] if len(coarse_values) > 1 else 1.0
    null_value = coarse_values[0] - coarse_width
    y_values = np.concatenate(([null_value], coarse_values))

    _save_numeric_heatmap(
        M,
        outpath=outpath,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        y_min=float(y_values[0]),
        y_max=float(y_values[-1]),
    )


def save_categorical_heatmap(qs_over_time, outpath, title):
    M = np.stack(qs_over_time, axis=1)
    _save_numeric_heatmap(
        M,
        outpath=outpath,
        title=title,
        xlabel="time",
        ylabel="state index",
        y_min=0.0,
        y_max=float(M.shape[0] - 1),
    )


def save_attention_line_plot(qs_over_time, outpath: Path, state_labels, title):
    M = np.stack(qs_over_time, axis=1)
    most_likely_state = np.argmax(M, axis=0)
    timesteps = np.arange(M.shape[1])

    outpath.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, most_likely_state, color="black", linewidth=2)
    plt.scatter(timesteps, most_likely_state, color="black", s=18)
    plt.xlabel("time")
    plt.ylabel("attention state")
    plt.yticks(np.arange(len(state_labels)), state_labels)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
