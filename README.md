<!-- These are examples of badges you might want to add to your README:
please update the URLs accordingly -->

[![Build Status](https://api.cirrus-ci.com/github/schwallergroup/chaos.svg?branch=main)](https://cirrus-ci.com/github/<USER>/additive-bo)
[![ReadTheDocs](https://readthedocs.org/projects/additive-bo/badge/?version=latest)](https://additive-bo.readthedocs.io/en/stable/)
[![Coveralls](https://img.shields.io/coveralls/github/<USER>/additive-bo/main.svg)](https://coveralls.io/r/<USER>/additive-bo)
[![PyPI-Server](https://img.shields.io/pypi/v/additive-bo.svg)](https://pypi.org/project/additive-bo/)
[![Conda-Forge](https://img.shields.io/conda/vn/conda-forge/additive-bo.svg)](https://anaconda.org/conda-forge/additive-bo)
[![Twitter](https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter)](https://twitter.com/additive-bo)
[![Project generated with PyScaffold](https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold)](https://pyscaffold.org/)

---

# CHAOS: Chemical additives optimization screening

**chaos** leverages Bayesian optimization to optimize additives for chemical reactions.

At the heart of CHAOS lies a robust Bayesian optimization engine. By harnessing the power of probabilistic modeling, we are able to efficiently search through the vast chemical space for optimal additives. Bayesian optimization balances exploration and exploitation, effectively narrowing down the vast chemical space to the most promising areas.

Buit on top of Gauche, it provides diverse compound and reaction representations together with custom Gaussian process kernel functions.



---

## 🚀 Installation

To install the package we recommend using a conda environment:

```bash
$ conda env create --file environment.yaml
pip install rxnfp --no-deps
```

## 💪 Getting Started
To configure an optimization routine check the config.yaml file, update parameters and use:
```python
python train_cli.py --config run_config.yaml
```
To run a wandb sweep use the parameters from sweep_config.yaml to initialize the sweep.

## 👋 Contributing

Contributions, whether filing an issue, making a pull request, or forking, are appreciated.

##  Attribution

### ⚖️ License

The code in this package is licensed under the MIT License.

### 📖 Citation

Citation goes here!

### 🛠️ Making Changes & Contributing

This project uses [pre-commit](https://pre-commit.com/), please make sure to install it before making any changes:




<p align="center">
  <img src="./docs/_static/additive-pipeline.png" width="100%">
</p>


<h1 align="center">
  Bayesian optimization for additives screening
</h1>
