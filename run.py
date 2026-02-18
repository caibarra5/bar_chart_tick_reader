# filename: run.py

import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt

from pymdp.agent import Agent

# -------------------------------------------------
# Imports (UPDATED MODEL)
# -------------------------------------------------
from module_aif_bar_reader.env import (
    BarChartEnv,
    NULL,
    BELOW,
    IN,
    ABOVE,
    NOT_CLOSE_AT_ALL,
    VERY_CLOSE,
)



from module_aif_bar_reader.generate_model_ABCD_params import (
    build_A,
    build_B,
    build_C,
    build_D,
    get_dimensions,
)

from module_aif_bar_reader.py_module_Agent_observation_capabilities import (
    run_bar_chart_full_pipeline
)
from module_aif_bar_reader.py_module_Agent_aif_capabilities import (
    image_interpretation_output_to_agent
)

# -------------------------------------------------
# Utilities
# -------------------------------------------------
def expectation_and_variance(p, values):
    p = np.asarray(p, dtype=float)
    mean = np.sum(p * values)
    var = np.sum(p * (values - mean)**2)
    return float(mean), float(var)


def entropy(p, eps=1e-16):
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1.0)
    return float(-np.sum(p * np.log(p)))

def check_probvec(p, name):
    p = np.asarray(p, dtype=float)
    if not (np.all(np.isfinite(p)) and abs(p.sum() - 1.0) < 1e-3):
        raise ValueError(f"Bad probvec {name}")

def onehot(i, n, dtype=np.float32):
    v = np.zeros(n, dtype=dtype)
    v[int(i)] = 1.0
    return v

# -------------------------------------------------
# Pretty printing
# -------------------------------------------------

ATTN_MEANINGS = {
    0: "query_bar1",
    1: "query_bar2",
    2: "report_avg",
}

def feedback_to_str(fb):
    if fb == NULL: return "NULL"
    if fb == NOT_CLOSE_AT_ALL: return "not_close"
    if fb == VERY_CLOSE: return "very_close"
    return str(fb)

def interpret_obs(obs):
    o0, o1, o2 = obs

    def region(x):
        if x == NULL:
            return "NULL"
        if x == BELOW:
            return "BELOW"
        if x == IN:
            return "IN"
        if x == ABOVE:
            return "ABOVE"
        return str(x)

    return (
        f"o0(bar1_query)={region(o0)}, "
        f"o1(bar2_query)={region(o1)}, "
        f"o2(feedback)={feedback_to_str(o2)}"
    )



# -------------------------------------------------
# Policy Builder (PARTITION MODEL)
# -------------------------------------------------

def build_policies(Ncoarse):

    policies = []
    n_factors = 5

    # query bar1
    for q in range(Ncoarse):
        pol = np.zeros((1, n_factors), dtype=int)
        pol[0, 2] = 0
        pol[0, 3] = q
        policies.append(pol)

    # query bar2
    for q in range(Ncoarse):
        pol = np.zeros((1, n_factors), dtype=int)
        pol[0, 2] = 1
        pol[0, 3] = q
        policies.append(pol)

    # report
    for rep in range(Ncoarse + 1):
        pol = np.zeros((1, n_factors), dtype=int)
        pol[0, 2] = 2
        pol[0, 4] = rep
        policies.append(pol)

    return policies



# -------------------------------------------------
# Plotting
# -------------------------------------------------
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

    # Build fine value grid (data units)
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

    # Preallocate list
    mean_dists = []

    for t in range(T):

        p1 = qs_bar1_over_time[t]
        p2 = qs_bar2_over_time[t]

        # Convolution (sum distribution)
        p_sum = np.convolve(p1, p2)

        # Build sum value grid
        sum_values = np.add.outer(values, values).ravel()

        # Because convolution produces sorted sums,
        # we instead build correct sum grid directly:
        sum_values = np.linspace(
            values[0] + values[0],
            values[-1] + values[-1],
            len(p_sum)
        )

        # Now scale support for mean
        mean_values = 0.5 * sum_values

        mean_dists.append(p_sum)

    M = np.stack(mean_dists, axis=1)

    outpath.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10,6))
    plt.imshow(
        M,
        aspect="auto",
        origin="lower",
        extent=[0, T, mean_values[0], mean_values[-1]],
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

    M = np.stack(qs_over_time, axis=1)  # shape: (Nfine, T)

    # Map fine indices → data values (bin centers)
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
    )
    plt.colorbar(label="probability")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()



# -------------------------------------------------
# MAIN
# -------------------------------------------------

def main():

    # -------------------------------------------------
    # 1) Perception pipeline
    # -------------------------------------------------
    image_path = "2_bar_chart_output/two_bar_chart.png"
    output_dir = Path("output_python_scripts/full_pipeline")
    output_dir.mkdir(parents=True, exist_ok=True)

    run_bar_chart_full_pipeline(
        image_path=image_path,
        output_dir=str(output_dir)
    )

    axes_and_bars_path = output_dir / "inferred_axes_and_bars.json"
    ocr_path = output_dir / "ocr_data.json"

    bar_values, tick_values, n_ticks, n_bars = \
        image_interpretation_output_to_agent(
            axes_and_bars_json_path=str(axes_and_bars_path),
            ocr_json_path=str(ocr_path),
        )

    print("Bar values:", bar_values)
    print("Tick values:", tick_values)

    # -------------------------------------------------
    # 2) Environment
    # -------------------------------------------------

    env = BarChartEnv(
        bar_heights_values=bar_values[:2],
        tick_values=tick_values,
        fine_bins_per_tick=10,
        coarse_bins_per_tick=3,
        seed=0,
    )

    # Build fine value grid (data units)
    fine_values = []
    for fine_idx in range(env.Nfine):
        tick = fine_idx // env.n_fine
        fine_within = fine_idx % env.n_fine

        lo = env.tick_values[tick]
        hi = env.tick_values[tick + 1]
        width = (hi - lo) / env.n_fine
        center_val = lo + (fine_within + 0.5) * width
        fine_values.append(center_val)

    fine_values = np.array(fine_values)


    print("True hidden state:", env.get_true_states())

    # -------------------------------------------------
    # 3) Generative model
    # -------------------------------------------------

    A = build_A(env.n_ticks, env.n_fine, env.n_coarse)
    print("A[0] shape:", A[0].shape)

    B = build_B(env.n_ticks, env.n_fine, env.n_coarse)
    C = build_C(env.n_ticks, env.n_fine, env.n_coarse)
    D = build_D(env.n_ticks, env.n_fine, env.n_coarse)

    policies = build_policies(env.Ncoarse)


    num_controls = [
        1,
        1,
        3,
        env.Ncoarse,        
        env.Ncoarse + 1
    ]


    control_fac_idx = [2, 3, 4]


    agent = Agent(
        A=A,
        B=B,
        C=C,
        D=D,
        policies=policies,
        policy_len=1,
        inference_horizon=2,  # important for binary search
        control_fac_idx=control_fac_idx,
        num_controls=num_controls,
        use_states_info_gain=True,
        use_utility=True,
        gamma=8.0,
        action_selection="deterministic",
    )

    # -------------------------------------------------
    # 4) Active Inference Loop
    # -------------------------------------------------

    T = 12
    model_obs = [NULL, NULL, NULL]

    print("\n=== ACTIVE INFERENCE ===")
    print("Initial obs:", interpret_obs(model_obs))

    # storage for heatmaps
    qs_over_time = [[] for _ in range(len(D))]

    log_rows = []

    for t in range(T):

        print(f"\n--- timestep {t} ---")
        print("Observation:", interpret_obs(model_obs))

        qs = agent.infer_states(model_obs)

        # --- Entropies ---
        entropy_bar1 = entropy(qs[0])
        entropy_bar2 = entropy(qs[1])

        # --- Expectations ---
        E_bar1, _ = expectation_and_variance(qs[0], fine_values)
        E_bar2, _ = expectation_and_variance(qs[1], fine_values)

        # --- Mean distribution via convolution ---
        p_sum = np.convolve(qs[0], qs[1])

        # Build sum value grid
        sum_values = np.linspace(
            fine_values[0] + fine_values[0],
            fine_values[-1] + fine_values[-1],
            len(p_sum)
        )

        mean_values = 0.5 * sum_values

        # Normalize (numerical safety)
        p_sum = p_sum / p_sum.sum()

        entropy_mean = entropy(p_sum)
        E_mean, Var_mean = expectation_and_variance(p_sum, mean_values)


        for i, q in enumerate(qs):
            check_probvec(q, f"qs[{i}]")

        agent.infer_policies()
        chosen = agent.sample_action()


        u_attn = int(chosen[2])
        u_param = int(chosen[3])
        u_report = int(chosen[4])

        log_rows.append({
            "timestep": t,
            "observation": interpret_obs(model_obs),
            "action_attention": ATTN_MEANINGS[u_attn],
            "action_query_param": u_param,
            "action_report_choice": u_report,
            "entropy_bar1": entropy_bar1,
            "entropy_bar2": entropy_bar2,
            "entropy_mean": entropy_mean,
            "E_bar1": E_bar1,
            "E_bar2": E_bar2,
            "E_mean": E_mean,
            "Var_mean": Var_mean,
        })


        print("Action:", ATTN_MEANINGS[u_attn], "param=", u_param)

        # Clamp control states
        qs[2] = onehot(u_attn, 3)
        qs[3] = onehot(u_param, len(qs[3]))
        agent.qs = qs

        for f in range(len(qs)):
            qs_over_time[f].append(qs[f].copy())


        print("Entropy bar1:", entropy(qs[0]))
        print("Entropy bar2:", entropy(qs[1]))

        # Save posterior plots
        save_probvec_png(
            qs[0],
            Path("posterior_plots/bar1") / f"t{t:03d}.png",
            f"q(bar1_fine) t={t}"
        )
        save_probvec_png(
            qs[1],
            Path("posterior_plots/bar2") / f"t{t:03d}.png",
            f"q(bar2_fine) t={t}"
        )


        # Step environment
        model_obs = env.step((u_attn, u_param, u_report))

    import pandas as pd

    df = pd.DataFrame(log_rows)

    table_path = Path("posterior_plots") / "trajectory_log.csv"
    df.to_csv(table_path, index=False)

    print("\nTrajectory table saved to:")
    print(table_path.resolve())


    # -------------------------------------------------
    # Save heatmaps
    # -------------------------------------------------

    HEATMAP_DIR = Path("posterior_plots/heatmaps")

    state_names = [
        "bar1_fine",
        "bar2_fine",
        "attention",
        "query_param",
        "report_choice",
    ]

    # Bar height states use value-mapped heatmap
    save_heatmap_time_state(
        qs_over_time[0],
        env,
        HEATMAP_DIR / "bar1_fine_time_heatmap.png",
        title="q(bar1_fine) over time",
    )

    save_heatmap_time_state(
        qs_over_time[1],
        env,
        HEATMAP_DIR / "bar2_fine_time_heatmap.png",
        title="q(bar2_fine) over time",
    )

    save_mean_heatmap(
        qs_over_time[0],
        qs_over_time[1],
        env,
        HEATMAP_DIR / "mean_height_time_heatmap.png",
    )


    # The remaining factors are categorical
    def save_categorical_heatmap(qs_over_time, outpath, title):
        M = np.stack(qs_over_time, axis=1)
        outpath.parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(8,5))
        plt.imshow(M, aspect="auto", origin="lower")
        plt.colorbar(label="probability")
        plt.xlabel("time")
        plt.ylabel("state index")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(outpath, dpi=200)
        plt.close()

    save_categorical_heatmap(
        qs_over_time[2],
        HEATMAP_DIR / "attention_time_heatmap.png",
        "q(attention) over time",
    )

    save_categorical_heatmap(
        qs_over_time[3],
        HEATMAP_DIR / "coarse_query_time_heatmap.png",
        "q(coarse_query) over time",
    )

    save_categorical_heatmap(
        qs_over_time[4],
        HEATMAP_DIR / "report_choice_time_heatmap.png",
        "q(report_choice) over time",
    )

    import pandas as pd

    df = pd.DataFrame(log_rows)

    TABLE_PATH = Path("posterior_plots") / "trajectory_log.csv"
    df.to_csv(TABLE_PATH, index=False)

    print("\nTrajectory table saved to:")
    print(TABLE_PATH.resolve())




    print("\nDone.")

    print("\nPosterior bar plots saved to:")
    print(Path("posterior_plots/bar1").resolve())
    print(Path("posterior_plots/bar2").resolve())

    print("\nHeatmaps saved to:")
    print(HEATMAP_DIR.resolve())


if __name__ == "__main__":
    main()
