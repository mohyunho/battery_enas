
#!/bin/bash
#PBS -l walltime=1:00:00
#PBS -j oe
#PBS -m abe
#PBS -M hyunho.mo@unitn.it
#PBS -o gpu.out
#PBS -q common_gpuQ
cd ~/home/hyunhomo/ieee-cim/neuroevolution-predictive-mainteinance/
source activate hmo
python3 launcher.py
conda deactivate
