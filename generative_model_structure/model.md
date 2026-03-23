# Bar Chart Active Inference Model

## 1. Description of Behavior

A person reading a bar chart will have an overview of all chart elements (bars, axes, tick marks, data values, bar labels, etc.), and thus can focus on their location in the chart. To read the chart, a person will note the bar label, look at the bar end, make a line from the bar top that projects onto the y-axis, note the data values of the tick-interval enclosing that point of projection, and estimate the data value that corresponds to this bar height.

A person estimates the data value for the bar height by drawing a line from the bar top to the y-axis. A human doesn't have pixel-level precision of the mapping from bar height to data value. This can be modeled by a Gaussian distribution, where the mean is the value the person chooses as the bar height. The variance could decrease with 1 or 2 more back-and-forth observations between the bar and the y-axis.

---

## 2. Environment

The environment extracts all the needed information from a simple, Python-generated PNG image of a bar graph. It does this using computer vision (OpenCV for geometric shapes and Tesseract for text).

The bar heights are read to pixel-level precision, and the data values are mapped to their midpoint in their bounding box and then to pixels in the image. The bar heights are determined by comparing their pixel heights to those of the data values.

This "ground truth" is not directly returned to the agent. Instead, observations are provided in coarse forms (coarse bins, tick intervals, and reported average feedback).

---

## 3. Hidden States

<!-- paste your Hidden States table here -->

---

## 4. Observations

<!-- paste your Observations table here -->

---

## 5. Actions

<!-- paste your Actions table here -->

---

## 6. Parametrization

### A. Likelihood Model

When `prev_attn = bar1`, the environment provides:
- `coarse_ruler_bar1`: correct coarse bin  
- `tick_interval_bar1`: correct tick interval  
- all other modalities: `"null"`

Only `bar1_height_fine` is affected.

Probability mass collapses to the correct tick interval, with fine bins following a discretized, truncated Gaussian. Probability accumulates near interval boundaries due to truncation. The likelihood matrix \(A\) reflects this behavior.

The same structure applies symmetrically for `bar2`.

---

### B. Transitions

All transition matrices are identity matrices.

Hidden states are static. No action (attention or reporting) changes the underlying bar graph.

---

### C. Preferences

The agent is indifferent to all modalities except `report_avg_feedback`.

Preferences:
- `null` → 0  
- `not_close` → -6.0  
- `close` → 2.0  
- `very_close` → 6.0  

All other modalities have uniform preference (0).

---

### D. Initial Beliefs

Beliefs over:
- `bar1_height_fine`
- `bar2_height_fine`

are uniform at time \(t = 0\).