
# OLMo-ladder

The OLMo-ladder is a set of scripts and model configurations for fitting scaling laws on smaller models, to take modeling or data-mixing decisions for pretraining. 

For details on reproducing results from the [paper](https://arxiv.org/pdf/2412.04403), please see [src/scripts/paper](src/scripts/paper/README.md).

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
Instructions for adding new evaluation sets can also be found there.


## Scaling laws for modeling decisions

Scaling laws can be fitted for taking modeling decisions, such as specific configuration changes, hyperparameter choices, etc.

* [WSD ladder](README.md)
* [muP ladder](README.md)

TODO: links

## Scaling laws for data decisions

TODO: add link to Ian's work.

## Miscellaneous

    Downloading W&B results.
    TODO: Add comet.


## Citation

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