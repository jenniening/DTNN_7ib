DTNN_7ib
==============================

Molecular energy and ligand stability prediction models based on deep neural tensor networks and MMFF optimized geometries.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile, used to create new environment, install requiremnts.
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data used to generate results in paper.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    ├── models             <- Trained models.
    ├── notebooks          <- Jupyter notebooks inlcuding tutorials.
    ├── reports            <- Generated analysis results.
    │   └── result_data    <- Result data for confs, used to get conformation stability result.
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── prepare_dataset.py
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


Detailed Information
------------
### Setup: <br>
Setup functions are in Makefile. To see functions in Makefile:
```
make -f Makefile
```
1. Create DTNN_7id environment:
```
make Makefile create_environment
```
2. Activate environment:
```
source activate DTNN_7id 
```
3. Install requirements:
```
make Makefile requirements
```
### Data Preparation: (src/data) <br>
All scipts for data prepartion are in src/data directory. To see the option for data prepation:
```
python prepare_dataset.py --help
```
To build qm9mmff dataset for training, validation, testlive and testing
```
python prepare_dataset.py
```
To build eMol9_C<sub>M</sub>:
```
python prepare_dataset.py --datatype emol9mmff
```
To build Plati_C<sub>M</sub>:
```
python prepare_dataset.py --datatype platinummmff
```
raw data are in data/raw directory, processed data are in data/processed directory. The processed data we used in data/external directory
### Model Training: (src/model) <br>
All scipts for model training are in src/data directory. To see the option for model training:
```
python train_model.py --help
```
Train DTNN_7id model:
```
python train_model.py --addnewm 
```
Train TL_QM9<sub>M</sub>:
```
python train_model.py --geometry MMFF --transferlearning
```
Train TL_eMol9_C<sub>M</sub>:
```
python train_model.py --datatype emol9mmff --geometry MMFF --transferlearning
```
### Prediction:(src/model) <br>
All scipts for prediction are in src/data directory. To see the option for prediction:
```
python predict_model.py --help
```
Performanc of DTNN_7id on QM9MMFF:
```
python predict_model.py 
```
Performance of TL_QM9<sub>M</sub> on QM9MMFF:
```
python predict_model.py --testpositions mmffpositions
```
Peformance of TL_eMol9_C<sub>M</sub> on eMol9_C<sub>M</sub>:
```
python predict_model.py --modelname TL_eMOL9_CM_name --testtype emol9mmff --testpositions positions1
```
Peformance of TL_eMol9_C<sub>M</sub> on Plati_C<sub>M</sub>:
```
python predict_model.py --modelname TL_eMOL9_CM_name --testtype platinummmff --testpositions positions1
```
Note: remember to change the model name based<br>
<br>
Thanks for DTNN code(https://github.com/atomistic-machine-learning/dtnn), we reimplemented elementary blocks in DTNN.

