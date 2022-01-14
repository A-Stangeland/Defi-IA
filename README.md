# Defi-IA

# Results
Our model achieved a MAPE score of 30.44090 on the private leaderboard of the kaggle competition, placing us 28th in the competition.
The instructions to reproduce our results are included below in the installation section.

# Installation
To install the required packages execute the following commands:
```bash
pip install -r requirements.txt
```

We recommend using conda to set up a virtual enviroment. To do this, execute the following commands:

```bash
conda create --name defi-ia python=3.9
conda activate defi-ia
pip install -r requirements.txt
```
# Usage
To train a model and evaluate it on the test data, run `train.py` by executing the following command:
```bash
python train.py
```
To generate the dataset from the kaggle data, run `data_processing.py`.
This script can take a long time to execute, so we have provided the already processed datasets.
When running `train.py`, you will be asked if you wish to download the datasets, which have a size around 65MB.
