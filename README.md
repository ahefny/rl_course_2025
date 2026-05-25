# Running Deep RL Experiments

## Option 1: Run Locally

### One-time Setup

Make sure Python, PIP and Git are installed.

#### [Optional] Creating and Activating a Python Virtual Environment

It is good to install the packages needed for experiments in a separate Python
virtual environment in order not to mess with your main Python installation

```sh
# Create a new environment in `.venv`.
# You can replace `.venv` with any path.
python3 -m venv .venv

# Activate environment
source .venv/bin/activate
```

#### Installing Packages

In an empty directory, run

```sh
git clone https://github.com/ahefny/rl_course_2025.git .
pip install -r deep_rl/requirements.txt
```

### Running Experiments

```sh
python deep_rl/dqn.py  # To run dqn.py
```

In another terminal run this command to view tensorboard. Open the browser on the output link.

```
tensorboard --logdir runs
```

## Option 2: Run in Colab

Create and run these cells

1. Copies the code to the notebook local directory. The `.py` can be edited in Colab.

```python
from IPython.display import clear_output

!mkdir -p rl_course_repo
!rm -rf rl_course_repo
!git clone https://github.com/ahefny/rl_course_2025.git rl_course_repo
!pip install -r rl_course_repo/deep_rl/requirements_colab.txt
!mv rl_course_repo/deep_rl/* .
!rm -rf rl_course_repo

clear_output()
```

1. Launches tensorboard

```python
%reload_ext tensorboard
%tensorboard --logdir runs
```

1. Runs experiment

```python
# Option 1: Copy the experiment code (e.g. `dqn.py`) to the cell and run.
# This keeps your edits if the Colab session is terminated and local files are lost.

# Option 2: Run the experiment as a shell command
!python3 dqn.py
```

