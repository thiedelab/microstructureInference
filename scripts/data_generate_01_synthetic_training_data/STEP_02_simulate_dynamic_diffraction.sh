#!/bin/bash

## Source /path/to/your/conda/installation/etc/profile.d/conda.sh
source /home/kwang/anaconda3/etc/profile.d/conda.sh

## Activate your desired Conda environment
conda activate py4DSTEM14.08-envforPlot

echo "starting at `date` on `hostname`"
echo ""
echo "You can split up this job into 19 different jobs to run simulation simultaenously"
echo ""
echo "simulating dynamic diffractions JOB (START)"
python step_02_simulate_from_ori_thick_symmOper.py --index 0
python step_02_simulate_from_ori_thick_symmOper.py --index 1
python step_02_simulate_from_ori_thick_symmOper.py --index 2
python step_02_simulate_from_ori_thick_symmOper.py --index 3
python step_02_simulate_from_ori_thick_symmOper.py --index 4
python step_02_simulate_from_ori_thick_symmOper.py --index 5
python step_02_simulate_from_ori_thick_symmOper.py --index 6
python step_02_simulate_from_ori_thick_symmOper.py --index 7
python step_02_simulate_from_ori_thick_symmOper.py --index 8
python step_02_simulate_from_ori_thick_symmOper.py --index 9
python step_02_simulate_from_ori_thick_symmOper.py --index 10
python step_02_simulate_from_ori_thick_symmOper.py --index 11
python step_02_simulate_from_ori_thick_symmOper.py --index 12
python step_02_simulate_from_ori_thick_symmOper.py --index 13
python step_02_simulate_from_ori_thick_symmOper.py --index 14
python step_02_simulate_from_ori_thick_symmOper.py --index 15
python step_02_simulate_from_ori_thick_symmOper.py --index 16
python step_02_simulate_from_ori_thick_symmOper.py --index 17
python step_02_simulate_from_ori_thick_symmOper.py --index 18
python step_02_simulate_from_ori_thick_symmOper.py --index 19


echo "ending at `date` on `hostname`"
echo "simulating dynamic diffractions JOB (END)"
