# filename: test/test_extract_chart_data_from_png.py

import sys
import math
import json
from pathlib import Path

import matplotlib.pyplot as plt

# -------------------------------------------------
# Allow imports from project root
# -------------------------------------------------
sys.path.append(str(Path(__file__).resolve().parents[1]))

from module_aif_bar_reader.py_module_Agent_observation_capabilities import (
    extract_chart_data_from_png
)

# -------------------------------------------------
# Output directories
# -------------------------------------------------
test_name = Path(__file__).stem
root_test_dir = Path("test") / test_name

images_dir = root_test_dir / "images"
outputs_dir = root_test_dir / "outputs"

images_dir.mkdir(parents=True, exist_ok=True)
outputs_dir.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------
# Helper: nice tick spacing
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
# Generate simple test chart
# -------------------------------------------------
def make_bar_chart_png(values, n_ticks, filepath):

    labels = ["A", "B"]

    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(labels, values)

    y_min = 0
    y_max = math.ceil(max(values) * 1.1)

    interval = nice_tick_interval(y_max, n_ticks)
    y_max = math.ceil(y_max / interval) * interval

    ax.set_ylim(y_min, y_max)

    ticks = list(range(0, y_max + interval, interval))
    ax.set_yticks(ticks)

    ax.set_ylabel("Data Value")
    ax.set_title("Two-Bar Chart")

    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()


# -------------------------------------------------
# Test cases
# -------------------------------------------------
TEST_CASES = [

    ([10, 20], 5),
    ([5, 30], 6),
    ([50, 80], 5),
    ([12, 14], 4),
    ([100, 250], 6),
    ([7, 62], 7),
]


# -------------------------------------------------
# Main test runner
# -------------------------------------------------
def main():

    print("\nRunning wrapper test\n")

    for i, (values, n_ticks) in enumerate(TEST_CASES):

        case_name = f"case_{i:03d}"

        img_path = images_dir / f"{case_name}.png"
        case_output_dir = outputs_dir / case_name

        print("\n----------------------------------")
        print("Case:", case_name)
        print("True values:", values)

        # generate PNG
        make_bar_chart_png(values, n_ticks, img_path)

        # run wrapper
        result = extract_chart_data_from_png(
            image_path=str(img_path),
            output_dir=str(case_output_dir)
        )

        print("Extracted data:")
        print(json.dumps(result, indent=2))

        chart_json = case_output_dir / "chart_data.json"

        print("Saved JSON:", chart_json.resolve())
        print("Image:", img_path.resolve())

    print("\nDone.")
    print("Images:", images_dir.resolve())
    print("Outputs:", outputs_dir.resolve())


# -------------------------------------------------

if __name__ == "__main__":
    main()