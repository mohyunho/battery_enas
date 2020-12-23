#!/bin/bash 
#PBS -l select=1:mem=128gb   
#PBS -l walltime=6:00:00
#PBS -j oe
#PBS -m abe
#PBS -M hyunho.mo@unitn.it
#PBS -o results_gpu.out
#PBS -q common_gpuQ
cd ~/home/hyunhomo/ieee-cim/neuroevolution-predictive-mainteinance/
source activate hmo
python3 launcher.py
conda deactivate
