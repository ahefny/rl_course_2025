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

- Open the [Colab notebook](https://colab.research.google.com/github/ahefny/rl_course_2025/blob/main/deep_rl/colab/notebook.ipynb).
- Replace the script with the one you would like to run
- Run the cells.

