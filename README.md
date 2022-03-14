# bsfit

author: steeve laquitaine

Go to [documentation](https://steevelaquitaine.github.io/bsfit/)

## Setup

Create virtual environment and install dependencies:

```bash
conda create -n heuristic python==3.6.13
conda activate heuristic
conda install --file src/requirements.txt -y
```

## Build & deploy docs

Basic steps: develop (features) -> pre-prod (homologation) -> master (production)

On branch `develop` (to add features):

1. Build docs:

```bash
cd docs/sphinx
make clean; make html # build html
```

2. Deploy docs:

```bash
cp -a docs/sphinx/_build/html/. docs/ # copy to deploy path
touch docs/.nojekyll # ignore jekyll on github
```

1. Pull request `pre-prod` branch (homologation to check docs rendering)

2. Pull request to `master` (production branch for end-users)


## Unit-testing

Unit-Test all the package's functions:

```bash
pytest src/test.py
```