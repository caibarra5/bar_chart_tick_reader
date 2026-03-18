from pathlib import Path

import numpy as np
import pandas as pd

from aif_bar_chart_reader.plotting import (
    save_categorical_heatmap,
    save_heatmap_time_state,
    save_mean_heatmap,
    save_probvec_png,
)


def run_active_inference_loop(
    agent,
    env,
    D,
    T,
    fine_values,
    posterior_dir,
    bar1_dir,
    bar2_dir,
    heatmap_dir,
    initial_obs,
    attn_meanings,
    interpret_obs,
    expectation_and_variance,
    entropy,
    check_probvec,
    onehot,
):
    model_obs = list(initial_obs)

    print("\n=== ACTIVE INFERENCE ===")
    print("Initial obs:", interpret_obs(model_obs))

    qs_over_time = [[] for _ in range(len(D))]
    log_rows = []

    for t in range(T):
        print(f"\n--- timestep {t} ---")
        print("Observation:", interpret_obs(model_obs))

        qs = agent.infer_states(model_obs)

        entropy_bar1 = entropy(qs[0])
        entropy_bar2 = entropy(qs[1])

        E_bar1, _ = expectation_and_variance(qs[0], fine_values)
        E_bar2, _ = expectation_and_variance(qs[1], fine_values)

        p_sum = np.convolve(qs[0], qs[1])
        sum_values = np.linspace(
            fine_values[0] + fine_values[0],
            fine_values[-1] + fine_values[-1],
            len(p_sum),
        )
        mean_values = 0.5 * sum_values

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

        log_rows.append(
            {
                "timestep": t,
                "observation": interpret_obs(model_obs),
                "action_attention": attn_meanings[u_attn],
                "action_query_param": u_param,
                "action_report_choice": u_report,
                "entropy_bar1": entropy_bar1,
                "entropy_bar2": entropy_bar2,
                "entropy_mean": entropy_mean,
                "E_bar1": E_bar1,
                "E_bar2": E_bar2,
                "E_mean": E_mean,
                "Var_mean": Var_mean,
            }
        )

        print("Action:", attn_meanings[u_attn], "param=", u_param)

        qs[2] = onehot(u_attn, 3)
        qs[3] = onehot(u_param, len(qs[3]))
        agent.qs = qs

        for f in range(len(qs)):
            qs_over_time[f].append(qs[f].copy())

        print("Entropy bar1:", entropy(qs[0]))
        print("Entropy bar2:", entropy(qs[1]))

        save_probvec_png(
            qs[0],
            bar1_dir / f"t{t:03d}.png",
            f"q(bar1_fine) t={t}",
        )
        save_probvec_png(
            qs[1],
            bar2_dir / f"t{t:03d}.png",
            f"q(bar2_fine) t={t}",
        )

        model_obs = env.step((u_attn, u_param, u_report))

    df = pd.DataFrame(log_rows)

    table_path = posterior_dir / "trajectory_log.csv"
    df.to_csv(table_path, index=False)

    print("\nTrajectory table saved to:")
    print(table_path.resolve())

    heatmap_dir = Path(heatmap_dir)

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

    return {
        "table_path": table_path,
        "heatmap_dir": heatmap_dir,
        "qs_over_time": qs_over_time,
        "log_rows": log_rows,
    }
