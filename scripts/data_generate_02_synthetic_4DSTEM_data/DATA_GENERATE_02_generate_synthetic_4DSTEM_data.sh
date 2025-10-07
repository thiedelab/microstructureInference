#!/bin/bash

## Source /path/to/your/conda/installation/etc/profile.d/conda.sh
source /home/kwang/anaconda3/etc/profile.d/conda.sh

## Activate your desired Conda environment
conda activate py4DSTEM14.08-envforPlot

echo "Make sure bash files in './scripts/data_generate_02_synthetic_4DSTEM_data' are executable"
echo "chmod +x ./scripts/data_generate_01_synthetic_training_data/STEP_01_sample_orientation_thickness.sh"
echo "chmod +x ./scripts/data_generate_01_synthetic_training_data/STEP_02_simulate_dynamic_diffraction.sh STEP_03_digitize_dynamic_diffraction.sh STEP_04_merge_and_split.sh"
echo "chmod +x ./scripts/data_generate_01_synthetic_training_data/STEP_03_digitize_dynamic_diffraction.sh STEP_04_merge_and_split.sh"
echo "chmod +x ./scripts/data_generate_01_synthetic_training_data/STEP_04_merge_and_split.sh"

echo "STEP_01_sample_orientation_thickness.sh"
./scripts/data_generate_01_synthetic_training_data/STEP_01_sample_orientation_thickness.sh

echo "STEP_02_simulate_dynamic_diffraction.sh"
./scripts/data_generate_01_synthetic_training_data/STEP_02_simulate_dynamic_diffraction.sh

echo "STEP_03_digitize_dynamic_diffraction.sh"
./scripts/data_generate_01_synthetic_training_data/STEP_03_digitize_dynamic_diffraction.sh

echo "STEP_04_merge_and_split.sh"
