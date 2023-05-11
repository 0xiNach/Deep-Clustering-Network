BD Computer Vision
==============================

Instructions
------------
1. Clone the repo.
2. *Optional:* Run `make virtualenv` to create a python virtual environment. Skip if using conda or some other env manager.
    1. Run `source env/bin/activate` to activate the virtualenv.
3. Install [Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli).
4. Run `make install-requirements` to install required python packages.
5. Run `make pre-commit-install` to install pre-commit hooks.
6. Run `python3 -m pip install .` to install cv_utils (utility scripts). Must run in root folder where *setup.py* is located
6. To create a new project, run `make create` and follow onscreen instructions.
7. To Configure SSH connection to Azure Repo, run `make configure-ssh` and follow on screen instructions. (Must required to run DVC)
8. *Optional:* Run `git add-connection` to add azure connection string to DVC config file, should run it manually after each push.


Project Organization
------------

    ├── LICENSE
    │
    ├── Makefile           <- Makefile with commands like `make install-requirements` or `make clean`
    │
    ├── README.md          <- The top-level README for developers using this project.
    │  
    ├── research  
    │   ├── image-classification
    │   │   └── project
    │   │       ├── .dvc  
    │   │       │   └── config         <- DVC configuration files
    │   │       │
    │   │       ├── data
    │   │       │   ├── processed      <- The final, canonical data sets for modeling.
    │   │       │   └── raw            <- The original, immutable data dump.
    │   │       │
    │   │       ├── models
    │   │       │   └── model.h5       <- Saved model file.
    │   │       │
    │   │       ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   │       │   └── figures        <- Generated graphics and figures to be used in reporting
    │   │       │   └── metrics.txt    <- Relevant metrics after evaluating the model.
    │   │       │   └── training_metrics.txt    <- Relevant metrics from training the model.
    │   │       │
    │   │       ├── src
    │   │       │   └── train.py       <- Model training script
    │   │       │   └── test.py        <- Model testing script
    │   │       │
    │   │       ├── dvc.lock           <- constructs the ML pipeline with defined stages.
    │   │       └── dvc.yaml           <- Training a model on the processed data.
    │   │
    │   ├── object-detection
    │   │   └── project                <- same directory structure as image-classification
    │   │
    │   └── computer-vision
    │       ├── project1
    │       └── project2
    │
    ├── cv_utils              <- global shareable python utility scripts  
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── .scripts           <- Contains bash helper scripts
    │
    ├── .github            <- Contains github actions configuration
    │
    ├── .vscode            <- vscode linter settings
    │
    ├── .pre-commit-config.yaml  <- pre-commit hooks file with selected hooks for the projects.
    │
    └── setup.py           <- makes cv_utils pip installable (pip install -e .) so cv_utils can be imported anywhere within python environment.  
