#!/bin/bash
#SBATCH --job-name="scikitlearn"
#SBATCH --time=0-00:5:00
#SBATCH --mail-user=mingbinc@student.unimelb.edu.au
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END

source /data/projects/punim0478/bryan/anaconda3/bin/activate gpu_env
python3 scikitlearn.py

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s

