This repository contains jupyter notebooks for developing self-supervised learning models for spectrum data processing.

## Setup
The notebooks and python files should be in a same folder. Model constants and paths which are in separate file **SSLConstants.py** should be adapted to the user working environment (models saving paths, loading paths, ect.)

## Experimenting
Dataset for training models and example model with corresonding statistics file are provided in the **shared/self-supervised-learning** folder on the [server](https://hub.over10k.ijs.si).
Training models and saving statictics data is performed with the **TrainScript.ipynb** python notebook. Check the comments in the notebook for more details.
Python notebook for model analysis, evaluation and visualization is provided as **ModelAnalysis.ipynb**.
All the necessary functions for both notebooks are contained in the **SSLUtils.py** file. In the current state, when working all the files should be in a same folder.

The project is in active development, so optimizations are lacking and feedback is welcomed.

## Related references
[Paper 1](https://arxiv.org/abs/2210.02899)
[Paper 2](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8977513)
[Testbed](https://log-a-tec.eu/datasets.html)
