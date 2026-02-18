import math
import yaml
from pathlib import Path
import matplotlib.pyplot as plt

# -------------------------------------------------
# Load config
# -------------------------------------------------
with open("aif_config.yaml", "r") as f:
    config = yaml.safe_load(f)

output_dir = config["output_dir"]
output_path = Path(output_dir)
output_path.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------
# Bar chart settings (modifiable)
# -------------------------------------------------
output_file = "two_bar_chart.png"  # change if desired

labels = ["A", "B"]
values = [28, 62]  # <-- change manually for testing
n_ticks = 5

# -------------------------------------------------
# Helper: human-friendly tick interval
# -------------------------------------------------
def nice_tick_interval(max_value, n_ticks):
    rough = max_value / (n_ticks - 1)
    magnitude = 10 ** math.floor(math.log10(rough))
    residual = rough / magnitude

    if residual < 1.5:
        nice = 1
    elif residual < 3:
        nice = 2
    elif residual < 7:
        nice = 5
    else:
        nice = 10

    return nice * magnitude


# -------------------------------------------------
# Create bar chart
# -------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(labels, values)

# Compute y-axis limits
y_min = 0
y_max = math.ceil(max(values) * 1.1)

interval = nice_tick_interval(y_max, n_ticks)
y_max = math.ceil(y_max / interval) * interval

ax.set_ylim(y_min, y_max)

tick_values = list(range(0, y_max + interval, interval))
ax.set_yticks(tick_values)

# Labels and title
ax.set_ylabel("Data Value")
ax.set_title("Two-Bar Chart")

# -------------------------------------------------
# Save figure
# -------------------------------------------------
save_path = output_path / output_file
plt.tight_layout()
plt.savefig(save_path, dpi=300)
plt.close()

print(f"Saved bar chart to: {save_path.resolve()}")
