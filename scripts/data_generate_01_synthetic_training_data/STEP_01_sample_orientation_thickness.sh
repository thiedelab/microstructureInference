#!/bin/bash

## Source /path/to/your/conda/installation/etc/profile.d/conda.sh
source /home/kwang/anaconda3/etc/profile.d/conda.sh

## Activate your desired Conda environment
conda activate py4DSTEM14.08-envforPlot


echo "starting at `date` on `hostname`"
echo ""
echo "SAMPING ORIENTATIONS and THICKNESS JOB (START)"

echo ""
echo "sampling orientations and thickness for copper (Cu) face-centered-cubic crystal"
python step_01_sample_orientation_thicnkess_mirrorOp.py --crystal Cu_fcc --outOfPlaneAngleDisp 2.0

echo ""
echo "sampling orientations and thickness for copper 1 oxide (Cu2O) cubic crystal"
python step_01_sample_orientation_thicnkess_mirrorOp.py --crystal Cu2O_cubic --outOfPlaneAngleDisp 2.0

echo ""
echo "sampling orientations and thickness for copper 1 oxide (Cu2O) cubic crystal"
python step_01_sample_orientation_thicnkess_mirrorOp.py --crystal CuO_monoclinic --outOfPlaneAngleDisp 4.5

echo ""
echo "SAMPING ORIENTATIONS and THICKNESS JOB (END)"
