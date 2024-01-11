#!/bin/bash

# Run the first Python command
python src/stages/data_prep.py --config=params.yaml

# Run the second Python command
python src/stages/data_split.py --config=params.yaml

# Run the third Python command
python src/stages/train_model.py --config=params.yaml

# Run the fourth Python command
python src/stages/evaluate.py --config=params.yaml