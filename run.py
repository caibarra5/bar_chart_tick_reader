# filename: run.py

import yaml
import numpy as np
from pathlib import Path
import json
import pandas as pd

from pymdp.agent import Agent

from aif_bar_chart_reader.analysis.plotting import (
    save_categorical_heatmap,
    save_heatmap_time_state,
    save_mean_heatmap,
    save_probvec_png,
)
from aif_bar_chart_reader.inference.agent_runner import (
    ABOVE,
    BELOW,
    IN,
    NOT_CLOSE_AT_ALL,
    NULL,
    VERY_CLOSE,
    BarChartEnv,
    build_A,
    build_B,
    build_C,
    build_D,
    get_dimensions,
    image_interpretation_output_to_agent,
    run_active_inference_loop,
    run_bar_chart_full_pipeline,
)
from aif_bar_chart_reader.inference.policies import build_policies


from aif_bar_chart_reader.analysis.metrics import (
    entropy,
    expectation_and_variance,
)

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
    o0, o1, o2, o3, o4 = obs
    return f"o0={o0}, o1={o1}, o2={o2}, o3={o3}, o4={o4}"


# -------------------------------------------------
# MAIN
# -------------------------------------------------

def main():

    # -------------------------------------------------
    # Load configuration
    # -------------------------------------------------
    with open("aif_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    image_path = config["bargraph_path"]
    fine_bins_per_tick = config["fine_bins_per_tick"]
    coarse_bins_per_tick = config["coarse_bins_per_tick"]
    inference_horizon = config["inference_horizon"]
    gamma = config["gamma"]
    T = config["T"]
    seed = config["random_seed"]

    output_root = Path(config["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)

    posterior_dir = output_root / "posterior_plots"
    bar1_dir = posterior_dir / "bar1"
    bar2_dir = posterior_dir / "bar2"
    heatmap_dir = posterior_dir / "heatmaps"

    for d in [posterior_dir, bar1_dir, bar2_dir, heatmap_dir]:
        d.mkdir(parents=True, exist_ok=True)


    # -------------------------------------------------
    # 1) Perception pipeline
    # -------------------------------------------------
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
        fine_bins_per_tick=fine_bins_per_tick,
        coarse_bins_per_tick=coarse_bins_per_tick,
        seed=seed,
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
        inference_horizon=inference_horizon,  # important for binary search
        control_fac_idx=control_fac_idx,
        num_controls=num_controls,
        use_states_info_gain=True,
        use_utility=True,
        gamma=gamma,
        action_selection="deterministic",
    )

    # -------------------------------------------------
    # 4) Active Inference Loop
    # -------------------------------------------------
    results = run_active_inference_loop(
        agent=agent,
        env=env,
        D=D,
        T=T,
        fine_values=fine_values,
        initial_obs=[NULL, NULL, NULL, 0, 0],
        attn_meanings=ATTN_MEANINGS,
        interpret_obs=interpret_obs,
        expectation_and_variance=expectation_and_variance,
        entropy=entropy,
    )

    qs_over_time = results["qs_over_time"]
    log_rows = results["log_rows"]

    for t, qs in enumerate(zip(qs_over_time[0], qs_over_time[1])):
        q_bar1, q_bar2 = qs
        save_probvec_png(
            q_bar1,
            bar1_dir / f"t{t:03d}.png",
            f"q(bar1_fine) t={t}",
        )
        save_probvec_png(
            q_bar2,
            bar2_dir / f"t{t:03d}.png",
            f"q(bar2_fine) t={t}",
        )

    df = pd.DataFrame(log_rows)
    table_path = posterior_dir / "trajectory_log.csv"
    df.to_csv(table_path, index=False)

    print("\nTrajectory table saved to:")
    print(table_path.resolve())

    save_heatmap_time_state(
        qs_over_time[0],
        env,
        heatmap_dir / "bar1_fine_time_heatmap.png",
        title="q(bar1_fine) over time",
    )

    save_heatmap_time_state(
        qs_over_time[1],
        env,
        heatmap_dir / "bar2_fine_time_heatmap.png",
        title="q(bar2_fine) over time",
    )

    save_mean_heatmap(
        qs_over_time[0],
        qs_over_time[1],
        env,
        heatmap_dir / "mean_height_time_heatmap.png",
    )

    save_categorical_heatmap(
        qs_over_time[2],
        heatmap_dir / "attention_time_heatmap.png",
        "q(attention) over time",
    )

    save_categorical_heatmap(
        qs_over_time[3],
        heatmap_dir / "coarse_query_time_heatmap.png",
        "q(coarse_query) over time",
    )

    save_categorical_heatmap(
        qs_over_time[4],
        heatmap_dir / "report_choice_time_heatmap.png",
        "q(report_choice) over time",
    )

    print("\nTrajectory table saved to:")
    print(table_path.resolve())

    print("\nDone.")

    print("\nPosterior bar plots saved to:")
    print(Path("posterior_plots/bar1").resolve())
    print(Path("posterior_plots/bar2").resolve())

    print("\nHeatmaps saved to:")
    print(heatmap_dir.resolve())


if __name__ == "__main__":
    main()
