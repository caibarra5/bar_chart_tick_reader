# Overview

Active inference (AIF) is a probabilistic modeling approach based on Bayesian inference, where an agent continuously updates beliefs about hidden states of the world using observations and selects actions that minimize expected uncertainty and maximize preferred outcomes.

In this repository, the code implements a pipeline that takes visual input (bar charts), processes it through structured modules, and applies inference mechanisms to extract meaningful information (e.g., values, structure, or relationships). The system separates data ingestion, inference, and output generation into distinct components, reflecting a layered design where perception, belief updating, and decision-like outputs are handled in a modular and interpretable way.

---

## Active Inference
The active inference defines three time series: hidden states s-tilde, observations o-tilde and actions u-tilde. Hidden states are regarded as the latent variables behind a generative process. The generative model tries to immitate the generative process. The hidden states are only partially observed, and thus only their probability distributions P(s) at each time step are computed. The observations are the variables emmited by the hidden states that transmit some information about the hidden states, particularly throught he likelihood P(o|s). Lastly, the actions are the 'active' part of active inference in which an active inference agetn interacts with the environment to influnced the hidden states. In practice, we try to fit the generative model by defining the hidden states factors, observation modalities, action factors and the states for each. 

## Purpose of Using Active inference
The purpose of using AIF in this project, is to simulate the observationa and inference capabilities of a human looking at a bar chart. In particular, we try to simulate how a human would extract all releveant information of a bar chart using the active inference framework. The hidden states in this application are considered the bar heights, and we restrict the AIF agent to having human level vision resolution. In this manner, the bar charts are read at a pixel level, but we simulate height reading to a more coarse level and with a truncated Gaussian distribution of the reading.

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
