## Hidden States

| Factor               | Factor States (Python Dictionary Representation)                                    | Notes |
|----------------------|-----------------------------------------------------------------------|------|
| bar1_height_fine     | {0: "fine_bin_0", 1: "fine_bin_1", ..., Nf-1: "fine_bin_Nf-1"}         | True fine height of bar 1. We don't use pixel-level resolution because that makes the computations longer (for state inference) and might be unnecessary, so we make it more coarse. |
| bar2_height_fine     | {0: "fine_bin_0", 1: "fine_bin_1", ..., Nf-1: "fine_bin_Nf-1"}         | True fine height of bar 2. It follows the same justification as bar1_height_fine. |
| attn                 | {0: "bar1", 1: "bar2", 2: "y-axis", 3: "feedback"}                     | Current (time t) focus of attention (either bar1, bar2, the y-axis, or the feedback when reporting an avg). It is a controlled state. |
| prev_attn            | {0: "bar1", 1: "bar2", 2: "y-axis", 3: "feedback"}                     | Current (time t) focus of attention (either bar1, bar2, the y-axis, or the feedback when reporting an avg). It is a controlled state. |
| avg_coarse_reported  | {0: "null", 1: "coarse_bin_0", ..., Nc: "coarse_bin_Nc-1"}             | The avg the agent reports. It is the midpoint of some coarse bin. The midpoint is given to the environment for some feedback. It also is a controlled state. |