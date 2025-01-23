
# OLMo-ladder

The OLMo-ladder is a set of scripts and model configurations for fitting scaling laws on smaller models, to take modeling or data-mixing decisions for pretraining.

## Table of Contents

- [Installation](#installation)
- [Running the ladder](#running-the-ladder)
- [Adding new evals](#adding-new-evals)
- [Scaling laws for modeling](#scaling-laws-for-modeling-eg-hyperparameters-modeling-config-changes-etc)
- [Scaling laws for data](#scaling-laws-for-data)
- [Miscellaneous](#miscellaneous)
- [Citation](#citation)



## Installation

```bash
conda create -n ladder python=3.10
conda activate ladder
cd OLMo-ladder
pip install -e ".[all]"  # options include plotting, beaker, wandb, dev, ladder
```

## Running the ladder

See [src/ladder](src/ladder/README.md) for instructions on running the ladder models.

## Adding new evals

    TODO: New named eval sets
    TODO: variance analysis when you add a new eval
    TODO: backfilling evals existing models

## Scaling laws for modeling (eg. hyperparameters, modeling config changes, etc.)

    TODO: adding new model configurations
    TODO: modeling code changes, eg. muP.
    TODO: add to examples folder?

## Scaling laws for data

    TODO: describe method
    TODO: add to examples folder

## Miscellaneous

    Downloading W&B results.
    TODO: Downloading comet results.


## Citation

For more details on reproducing results from the [paper](https://arxiv.org/pdf/2412.04403), please see [src/scripts/paper](src/scripts/paper/README.md).

```
@article{Bhagia2024EstablishingTS,
  title={Establishing Task Scaling Laws via Compute-Efficient Model Ladders},
  author={Akshita Bhagia and Jiacheng Liu and Alexander Wettig and David Heineman and Oyvind Tafjord and A. Jha and Luca Soldaini and Noah A. Smith and Dirk Groeneveld and Pang Wei Koh and Jesse Dodge and Hanna Hajishirzi},
  journal={ArXiv},
  year={2024},
  volume={abs/2412.04403},
  url={https://api.semanticscholar.org/CorpusID:274514987}
}
```