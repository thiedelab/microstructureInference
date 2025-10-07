#!/bin/bash

## Source /path/to/your/conda/installation/etc/profile.d/conda.sh
source /home/kwang/anaconda3/etc/profile.d/conda.sh

## Activate your desired Conda environment
conda activate py4DSTEM14.08-envforPlot

echo "starting at `date` on `hostname`"
echo ""
echo "digitizing dynamic diffractions JOB (START)"
python step_03_digitize_BraggDisks_in_each_simulated_pattern.py --index 0
python step_03_digitize_BraggDisks_in_each_simulated_pattern.py --index 1
python step_03_digitize_BraggDisks_in_each_simulated_pattern.py --index 2
python step_03_digitize_BraggDisks_in_each_simulated_pattern.py --index 3
python step_03_digitize_BraggDisks_in_each_simulated_pattern.py --index 4
python step_03_digitize_BraggDisks_in_each_simulated_pattern.py --index 5
python step_03_digitize_BraggDisks_in_each_simulated_pattern.py --index 6
python step_03_digitize_BraggDisks_in_each_simulated_pattern.py --index 7
python step_03_digitize_BraggDisks_in_each_simulated_pattern.py --index 8
python step_03_digitize_BraggDisks_in_each_simulated_pattern.py --index 9
python step_03_digitize_BraggDisks_in_each_simulated_pattern.py --index 10
python step_03_digitize_BraggDisks_in_each_simulated_pattern.py --index 11
python step_03_digitize_BraggDisks_in_each_simulated_pattern.py --index 12
python step_03_digitize_BraggDisks_in_each_simulated_pattern.py --index 13
python step_03_digitize_BraggDisks_in_each_simulated_pattern.py --index 14
python step_03_digitize_BraggDisks_in_each_simulated_pattern.py --index 15
python step_03_digitize_BraggDisks_in_each_simulated_pattern.py --index 16
python step_03_digitize_BraggDisks_in_each_simulated_pattern.py --index 17
python step_03_digitize_BraggDisks_in_each_simulated_pattern.py --index 18
python step_03_digitize_BraggDisks_in_each_simulated_pattern.py --index 19


echo "ending at `date` on `hostname`"
echo "digitizing dynamic diffractions JOB (END)"
