
#!/bin/bash  
#PBS -l select=1:mem=150gb     
#PBS -l walltime=72:00:00
#PBS -j oe
#PBS -m abe
#PBS -M hyunho.mo@unitn.it
#PBS -o hpc_score_3_results.out
#PBS -q common_gpuQ
cd ~/home/hyunhomo/ieee-cim/neuroevolution-predictive-mainteinance/
source activate hmo
python3 launcher2.py
conda deactivate
