# bsfit

author: steeve laquitaine

Go to [documentation](https://steevelaquitaine.github.io/bsfit/)

## Setup

Create virtual environment and install dependencies:

```bash
conda create -n bsfit python==3.6.13
conda activate bsfit
conda install --file bsfit/requirements.txt -y
```

## Tutorials

Setup jupyter notebook:

```bash
conda install -n heuristic ipykernel --update-deps --force-reinstall
```

## Contribute

### Unit-testing

Unit-Test all the package's functions:

```bash
pytest bsfit/test.py
```

### Documentation

#### Best practices:

- Keep Doctstrings in Google Style Guide format.
  
#### Update

1. Edit docs/source/ 
2. Go to "Build & deploy" section

#### Build & deploy

Basic steps: develop (features) -> pre-prod (homologation) -> master (production)

1. On branch `develop` (to add features):
    - when: after having added a new working feature
    
    ```bash
    bash deploy.sh
    ```

2. Pull request `pre-prod` branch (homologation, check docs rendering)
    - when: after deploying sphinx/ to docs/

3. Pull request to `master` (production branch for end-users)
    - when: the entire codebase is clean