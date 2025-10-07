#!/bin/bash

## Source /path/to/your/conda/installation/etc/profile.d/conda.sh
source /home/kwang/anaconda3/etc/profile.d/conda.sh

## Activate your desired Conda environment
conda activate py4DSTEM14.08-envforPlot


echo "Running step_01_sample_backgrounds_and_create_multiGrainScanSpace.py"
python step_01_sample_backgrounds_and_create_multiGrainScanSpace.py

echo "Running step_02_sample_orientations.py"
python step_02_sample_orientations.py

echo "Running step_03_generate_synthetic_4DSTEM_data_single_crystal.py"
python step_03_generate_synthetic_4DSTEM_data_single_crystal.py

echo "Running step_04_generate_synthetic_4DSTEM_data_multi_crystals.py"
python step_04_generate_synthetic_4DSTEM_data_multi_crystals.py
