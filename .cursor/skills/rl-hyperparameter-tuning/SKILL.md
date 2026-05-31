# RL Hyperparameter Tuning Skill

Use this skill to tune the hyperparameter of RL algorithm by modifying the config.

## Setting Up
- Unless the user provides a path to run the experiment, suggest an experiment name and set the experiment path to `./hp_tuning/{suggested_experiment_name}`. Use the path determined by the step to replace `{experimental_path}` in all steps below.
- Run `mkdir -p {experiment_path}`. Inform the user that you will be using this path for the experiment.
- Generate a `{experiment_path}/plan.md` file that includes a description of the experimental setup (e.g. the RL algorithm and environments), the optimization goal and termination condition, and the metrics to be tracked to guide fine tuning.

## Running an experiment
- Generate an id string in the format `{current timestamp formatted YY_MM_DD__hh_mm_ss}_{random string of length 5}`. Use it to replace `{run_id_string}` below.
- Generate a config file `{experiment_path}/{run_id_string}/config_override.yaml` based on recommendations from previous iteration. If this is the first iteration, generate a config file that only overrides the number of training steps to {training_steps}.
- You can run an experiment using `python deep_rl/sac.py --config {experiment_path}/{run_id_string}/config_override.yaml --root-log-dir {experiment_path} --run-name {run_id_string} 2>&1 > {experiment_path}/{run_id_string}/log.txt`.
- The file `{experiment_path}/{run_id_string}/log.txt` contains different loss curves in text format.
- Analyze the log file and provide a Markdown analysis file `{experiment_path}/{run_id_string}/analysis.md` that includes:
  - Learning problems with explanation including supporting evidence from the data.
  - Hyperparameter tuning suggestions. Include the rationale behind each suggestion. Do **NOT** recommend changing the total number of training steps. Your suggestions must take into account the analyses of all previous experiments (all `analysis.md` files in the current `{experiment_path}`).
  - A short summry of problems and suggested improvements.

Analyze the file and summarize the learning issues and suggest hyperparameter improvements. Include the rationale behind each suggestion. Output the summary and improvements in `{experiment_path}/{run_id_string}/analysis.txt`. Do not recmmend changing the total number of steps.

## Important Instructions
- DO **NOT** MODIFY any *.py file.
- **ALWAYS** use .venv python envrionment.
