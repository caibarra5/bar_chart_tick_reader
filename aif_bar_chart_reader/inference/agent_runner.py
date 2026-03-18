import numpy as np

from aif_bar_chart_reader.model.env import (
    ABOVE,
    BELOW,
    IN,
    NOT_CLOSE_AT_ALL,
    NULL,
    VERY_CLOSE,
    BarChartEnv,
    image_interpretation_output_to_agent,
    run_bar_chart_full_pipeline,
)
from aif_bar_chart_reader.model.generate_model_ABCD_params import (
    build_A,
    build_B,
    build_C,
    build_D,
    get_dimensions,
)


def check_probvec(p, name):
    p = np.asarray(p, dtype=float)
    if not (np.all(np.isfinite(p)) and abs(p.sum() - 1.0) < 1e-3):
        raise ValueError(f"Bad probvec {name}")


def onehot(i, n, dtype=np.float32):
    v = np.zeros(n, dtype=dtype)
    v[int(i)] = 1.0
    return v


def run_active_inference_loop(
    agent,
    env,
    D,
    T,
    fine_values,
    initial_obs,
    attn_meanings,
    interpret_obs,
    expectation_and_variance,
    entropy,
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

        model_obs = env.step((u_attn, u_param, u_report))

    return {
        "qs_over_time": qs_over_time,
        "log_rows": log_rows,
    }
