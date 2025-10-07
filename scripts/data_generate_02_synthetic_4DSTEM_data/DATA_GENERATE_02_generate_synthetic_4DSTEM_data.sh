#!/bin/bash

## Source /path/to/your/conda/installation/etc/profile.d/conda.sh
source /home/kwang/anaconda3/etc/profile.d/conda.sh

## Activate your desired Conda environment
conda activate py4DSTEM14.08-envforPlot

number_of_grains=60
scanSpaceDimensionForSynthetic4DATMdata=128

echo "Running step_01_sample_backgrounds_and_create_multiGrainScanSpace.py"
echo ""
python step_01_sample_backgrounds_and_create_multiGrainScanSpace.py --num_grains "$number_of_grains" --ScanDimension "$scanSpaceDimensionForSynthetic4DATMdata"

echo "Running step_02_sample_orientations.py"
echo ""
python step_02_sample_orientations.py --num_orientations "$number_of_grains"

echo "Running step_03_generate_synthetic_4DSTEM_data_single_crystal.py"
echo ""
python step_03_generate_synthetic_4DSTEM_data_single_crystal.py --num_grains "$number_of_grains" --ScanDimension "$scanSpaceDimensionForSynthetic4DATMdata" --num_orientations "$number_of_grains"

echo "Running step_04_generate_synthetic_4DSTEM_data_multi_crystals.py"
echo ""
python step_04_generate_synthetic_4DSTEM_data_multi_crystals.py --num_grains "$number_of_grains" --ScanDimension "$scanSpaceDimensionForSynthetic4DATMdata" --num_orientations "$number_of_grains"
