#!/bin/bash
#SBATCH --job-name="scikitlearn"
#SBATCH --time=0-00:5:00

module purge
module load foss/2022a
module load scikit-learn/1.1.2
module load matplotlib/3.5.2

python3 scikitlearn.py

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s

