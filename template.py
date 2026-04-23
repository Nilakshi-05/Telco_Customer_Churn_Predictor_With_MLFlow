import os
from pathlib import Path
import logging

logging.basicConfig(level = logging.INFO, format = "[%(asctime)s] : %(message)s:")

project_name = "mlProject"

list_of_files = [
    #DATA folders
    "data/raw/.gitkeep",
    "data/processed/.gitkeep",
    "data/external/.gitkeep",

    #Notebooks
    "notebooks/.gitkeep",

    #Source Code
    "src/data/__init__.py",
    "src/features/__init__.py",
    "src/models/__init__.py",
    "src/utils/__init__.py",

    #APP
    "app/__init__.py",
    "app/main.py",

    #Configs
    "configs/config.yaml",

    #Scripts
    "scripts/train.py",
    "scripts/predict.py",

    #Github Actions
    ".github/workflow/.gitkeep",

    #Docker
    "docker/Dockerfile",

    #Data Validation
    "great_expectations/.gitkeep",

    #MLflow
    "mlruns/.gitkeep",

    #Artifacts
    "artifacts/.gitkeep",

    #Root files
    "requirements.txt"
]

for path in list_of_files:
    path = Path(path)
    dir_name, file_name = os.path.split(path)

    if dir_name != "":
        os.makedirs(dir_name, exist_ok = True)
        logging.info(f"Created directory: {dir_name}")

        #Create empty file if it does not exist
        if(not os.path.exists(path)) or(os.path.getsize(path)==0):
            with open(path, "w") as f:
                pass
            logging.info(f"Created files: {path}")
        else:
            logging.info(f"{file_name} already exists")
