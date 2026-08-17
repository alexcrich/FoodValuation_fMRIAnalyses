#!/bin/bash
#SBATCH -J nilearn1stlevel_task_BI #name of experiment
#SBATCH --output=spm.txt
#SBATCH --error=spm.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=64G
#SBATCH -t 24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=a.rich@yale.edu

set -euo pipefail

module --force purge
module load Python/3.12.3-GCCcore-13.3.0
source ~/venv/foodvaluation/bin/activate
~/venv/foodvaluation/bin/pip install --quiet joblib nilearn

module load FSL/6.0.7.9
source $FSLDIR/etc/fslconf/fsl.sh
unset PYTHONPATH
unset PYTHONHOME

module load MATLAB/2023b

~/venv/foodvaluation/bin/python -c "import sys; print(sys.executable)"
~/venv/foodvaluation/bin/python -c "import nipype; print('nipype ok', nipype.__version__)"

~/venv/foodvaluation/bin/python /nfs/roberts/project/pi_il77/acr69/firstlevel_task_BI_nilearn.py > TaskBI.log