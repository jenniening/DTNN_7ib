DTNN_7id
==============================

This is a molecular energy and ligand stability prediction model based on deep neural tensor networks and MMFF optimized geometries

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile, used to create new environment, install requiremnts
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data used to generate results in paper
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    ├── models             <- Trained and serialized models, model predictions
    ├── notebooks          <- Jupyter notebooks inlcuding tutorials
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── result_data    <- Result data for confs.
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


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
