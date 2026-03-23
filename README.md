## Overview

AIF (Active Inference Framework) is a probabilistic modeling approach based on Bayesian inference, where an agent continuously updates beliefs about hidden states of the world using observations and selects actions that minimize expected uncertainty and maximize preferred outcomes.

In this repository, the code implements a pipeline that takes visual input (bar charts), processes it through structured modules, and applies inference mechanisms to extract meaningful information (e.g., values, structure, or relationships). The system separates data ingestion, inference, and output generation into distinct components, reflecting a layered design where perception, belief updating, and decision-like outputs are handled in a modular and interpretable way.

---

## How to Run

### 1. Configure Parameters

Edit the configuration file:

```bash
aif_config.yaml
```

### 2. Generate Test Bar Graph

```bash
python two_bar_png_file_generator.py
```

### 3. Run Environment, Agent and Inference
```bash
python run.py
```

### Alternative: Full Pipeline
Run everything (generate image + inference):
```bash
python full_pipeline_run.py
```

## Project Structure
The module for AIF inference is located in:
```bash
aif_bar_chart_reader/
```
The layer view png is located in:
```bash
assets/
```
The generative model structure is located in: 
```bash
generative_model_structure/
```
The testing of parts of the code is located in:
```bash
test/
```
