# bsfit

author: steeve laquitaine

Go to [documentation](https://inference-org.github.io/bsfit/)

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


## References

Main paper to be cited ([Laquitaine et al., 2018](https://reader.elsevier.com/reader/sd/pii/S0896627317311340?token=3C565810A1E5E3A3F8526045212D4915DAF3F6DE16339366119B4CF6B1D05FB762927F31382226BD199E132C0FAE216A&originRegion=eu-west-1&originCreation=20220331151804))

```
@article{laquitaine2018switching,
  title={A switching observer for human perceptual estimation},
  author={Laquitaine, Steeve and Gardner, Justin L},
  journal={Neuron},
  volume={97},
  number={2},
  pages={462--474},
  year={2018},
  publisher={Elsevier}
}
```
