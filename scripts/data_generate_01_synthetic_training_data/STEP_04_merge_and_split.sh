#!/bin/bash

## Source /path/to/your/conda/installation/etc/profile.d/conda.sh
source /home/kwang/anaconda3/etc/profile.d/conda.sh

## Activate your desired Conda environment
conda activate py4DSTEM14.08-envforPlot

echo "starting at `date` on `hostname`"
echo ""
echo "YOU MAY NEED LARGE RAM MEMORY FOR MERGING DATASETS. IF NECESSARY, PLEASE RUN THIS CODE IN HPC CLUSTER" 
echo ""
echo "merging and spliting data JOB (START)"
python step_04_merge_data_and_split_into_train_and_valid.py

echo "ending at `date` on `hostname`"
echo "merging and spliting data JOB (END)"
