# Renormalising Generative Models

This repository implements Renormalising Generative Models as described in https://arxiv.org/abs/2407.20292 using PYMDP.

In addition, we do some benchmarks on ML environments such as
- Atari Learning Environment
- DeepMind Control suite

## Installation

Install locally using pip, preferably in a separate Python virtual environment:

```
git clone git@github.com:VersesTech/rgm.git
cd rgm
pip install -e .
```

## Usage

Examples Jupyter notebooks are available in the `notebooks` folder.


## DeepMind Control Suite

To run the DeepMind Control Suite example, extra dependencies are required:

```
pip install -e .[dmc]
```

The `dm_control.ipynb` notebook demonstrates fitting an RGM on DMC expert data.


## Atari

To run the Atari examples, extra dependencies are required:

```
pip install -e .[atari]
```

To get expert Atari play data, download from https://drive.google.com/drive/folders/1JzF74ll6vpKgDs5jajcq3Xd2TQ2Kl-hE and put it in the `data/atari/<game>` folder.

Play around with the `atari.ipynb` notebook or run `python scripts/atari.py --game=boxing`