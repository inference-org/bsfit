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

1. On branch `develop` (to add features):
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

2. Pull request `pre-prod` branch (homologation, check docs rendering)
    - when: after deploying sphinx/ to docs/

3. Pull request to `master` (production branch for end-users)
    - when: the entire codebase is clean

## Unit-testing

Unit-Test all the package's functions:

```bash
pytest src/test.py
```