## Running Experiments

### Option 1: Run Locally

#### One time setup

Make sure Python, PIP and Git are installed. It is good to run experiments
in a Python environment if you do not want to mess with your main Python installation.

In an empty directory, run
```sh
git clone https://github.com/ahefny/rl_course_2025.git .
pip install -r requirements.txt
```

#### To run an experiment

```sh
python experiments/dqn.py  # To run dqn.py
```

In another terminal run this command to view tensorboard. Open the browser on the output link.
```
tensorboard --logdir runs
```

### Option 2: Run in Colab

Create and run these cells

1. Copies the code to the notebook local directory. The `.py` can be edited in Colab.

```python
from IPython.display import clear_output

!mkdir -p rl_course_repo
!rm -rf rl_course_repo
!git clone https://github.com/ahefny/rl_course_2025.git rl_course_repo
!pip install -r rl_course_repo/requirements_colab.txt
!mv rl_course_repo/experiments/* .
!rm -rf rl_course_repo

clear_output()
```

2. Launches tensorboard

```python
%reload_ext tensorboard
%tensorboard --logdir runs
```

3. Runs experiment
```python
# Option 1: Copy the experiment code (e.g. `dqn.py`) to the cell and run.
# This keeps your edits if the Colab session is terminated and local files are lost.

# Option 2: Run this
from importlib import reload
import dqn as experiment
reload(experiment)
experiment.main()
```



