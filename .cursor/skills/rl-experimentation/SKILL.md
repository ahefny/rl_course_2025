# RL Experimentaiton Skill

## Running an experiment
- Generate an id string in the format `{current timestamp formatted YY_MM_DD__hh_mm_ss}_{random string of length 5}`. Use it to replace `{id_string}` below.
- Generate a config file `logs/{id_string}_config.yaml` based on recommendations from previous iteration. If this is the first iteration, copy `configs/sac_default.yaml`
- You can run an experiment using `python experiments/sac.py --config logs/{id_string}_config.yaml > logs/{id_string}_log.txt`.
- The file `{id_string}_log.txt` contains different loss curves in text format.
- Analyze the file and summarize the learning issues and suggest hyperparameter improvements. Include the rationale behind each suggestion. Output the summary and improvements in `logs/{id_string}_analysis.txt`. Do not recmmend changing the total number of steps.

## Important Instructions
- DO **NOT** MODIFY any *.py file.
- **ALWAYS** use .venv python envrionment.
