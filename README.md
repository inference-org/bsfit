# heuristic

author: steeve laquitaine

## Setup

Create virtual environment and install dependencies:

```bash
conda create -n heuristic python==3.6.13
conda activate heuristic
conda install --file src/requirements.txt -y
```

## Unit-testing

Unit-Test all the package's functions:

```bash
pytest src/test.py
```