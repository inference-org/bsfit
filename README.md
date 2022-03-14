# heuristic

author: steeve laquitaine

## Setup

Create virtual environment and install dependencies:

```bash
conda create -n heuristic python==3.6.13
conda activate heuristic
conda install --file src/requirements.txt -y
```

## Build & deploy docs

On branch `develop` (to add features):

1. Build docs:

```bash
cd heuristic/sphinx
make clean; make html # build html
```

2. Deploy docs:

```bash
cp -a heuristic/sphinx/_build/html/. heuristic/docs/ # copy to deploy path
touch docs/.nojekyll # ignore jekyll on github
```

3. Merge changes to `master-docs` branch (for homologation)



## Unit-testing

Unit-Test all the package's functions:

```bash
pytest src/test.py
```