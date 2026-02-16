# filename: env.py
import numpy as np

NULL = 0
IN_BIN = 1
OUT_BIN = 2

NOT_CLOSE_AT_ALL = 1
CLOSE = 2
VERY_CLOSE = 3


class BarChartEnv:

    def __init__(
        self,
        bar_heights_values,
        tick_values,
        fine_bins_per_tick=10,
        coarse_bins_per_tick=3,
        seed=None,
    ):

        self.bar_values = [float(bar_heights_values[0]),
                           float(bar_heights_values[1])]

        self.tick_values = np.array(sorted(tick_values), dtype=float)
        self.n_ticks = len(self.tick_values) - 1

        self.n_fine = fine_bins_per_tick
        self.n_coarse = coarse_bins_per_tick

        self.Nfine = self.n_ticks * self.n_fine
        self.Ncoarse = self.n_ticks * self.n_coarse

        self.rng = np.random.default_rng(seed)

        self.bar1_fine_true = self._value_to_fine_global(self.bar_values[0])
        self.bar2_fine_true = self._value_to_fine_global(self.bar_values[1])

        self.true_avg_coarse_global = self._true_avg_coarse()

    # --------------------------------------------------------
    # Discretization
    # --------------------------------------------------------

    def _value_to_tick_idx(self, value):
        idx = int(np.searchsorted(self.tick_values, value, side="right") - 1)
        return int(np.clip(idx, 0, self.n_ticks - 1))

    def _value_to_fine_global(self, value):
        tick_idx = self._value_to_tick_idx(value)
        lo = self.tick_values[tick_idx]
        hi = self.tick_values[tick_idx + 1]
        frac = 0.0 if hi <= lo else (value - lo) / (hi - lo)
        frac = np.clip(frac, 0.0, 1.0)
        fine_within = int(np.floor(frac * self.n_fine))
        fine_within = np.clip(fine_within, 0, self.n_fine - 1)
        return tick_idx * self.n_fine + fine_within

    def _fine_to_coarse(self, fine):
        tick = fine // self.n_fine
        fine_within = fine % self.n_fine
        coarse_within = (fine_within * self.n_coarse) // self.n_fine
        return tick * self.n_coarse + coarse_within

    def _true_avg_coarse(self):
        avg = int(np.floor(0.5 * (self.bar1_fine_true +
                                  self.bar2_fine_true)))
        return self._fine_to_coarse(avg)

    # --------------------------------------------------------
    # Interaction
    # --------------------------------------------------------

    def fine_global_to_value(self, fine_idx):
        tick = fine_idx // self.n_fine
        fine_within = fine_idx % self.n_fine

        lo = self.tick_values[tick]
        hi = self.tick_values[tick + 1]

        width = (hi - lo) / self.n_fine
        return lo + (fine_within + 0.5) * width


    def step(self, action):

        u_attn = int(action[0])
        u_query = int(action[1])
        u_report = int(action[2])

        o0 = NULL
        o1 = NULL
        o2 = NULL

        # bar1 coarse bin query
        if u_attn == 0:
            true_bin = self._fine_to_coarse(self.bar1_fine_true)
            if u_query == true_bin:
                o0 = IN_BIN
            else:
                o0 = OUT_BIN

        # bar2 coarse bin query
        elif u_attn == 1:
            true_bin = self._fine_to_coarse(self.bar2_fine_true)
            if u_query == true_bin:
                o1 = IN_BIN
            else:
                o1 = OUT_BIN

        # report
        elif u_attn == 2:
            if u_report == 0:
                o2 = NULL
            else:
                report_bin = u_report - 1
                if report_bin == self.true_avg_coarse_global:
                    o2 = VERY_CLOSE
                else:
                    o2 = NOT_CLOSE_AT_ALL

        return [o0, o1, o2]

    def get_true_states(self):
        return {
            "bar1_fine_true": int(self.bar1_fine_true),
            "bar2_fine_true": int(self.bar2_fine_true),
            "true_avg_coarse_global": int(self.true_avg_coarse_global),
        }
