#!/bin/bash  
#PBS -l select=1:mem=100gb   
#PBS -l walltime=2:00:00
#PBS -j oe
#PBS -m abe
#PBS -M hyunho.mo@unitn.it
#PBS -o hpc_multigpu.out
#PBS -q common_gpuQ
cd ~/home/hyunhomo/ieee-cim/neuroevolution-predictive-mainteinance/
source activate hmo
python3 launcher.py
conda deactivate
