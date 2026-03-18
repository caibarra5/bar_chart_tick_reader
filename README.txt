
First look choose the parameters in the configuraiton file:
aif_config.yaml

Second, run the test bar graph generator:
python two_bar_png_file_generator.py
The two bar heights can be specified in the config file.

Then run the env generator, agent and inference loop:
python run.py

Alternatively, one can run full_pipeline_run.py to both create the png file reason about the bar graph:
python full_pipeline_run.py

The module for aif inference is aif_bar_chart_reader/
