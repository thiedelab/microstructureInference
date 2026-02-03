#!/bin/bash

source /path/to/new/virtual/environment/bin/activate

export OMP_NUM_THREADS=1

echo "starting at `date` on `hostname`"
echo "SLURM_JOB_ID = $SLURM_JOB_ID"

python3 train_for_orientation_prediction.py --initial_run --isMultitask 0

echo "JOB DONE"

