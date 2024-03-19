#!/bin/bash
#SBATCH --partition=gpu-a100-short
#SBATCH --qos=gpgpudeeplearn
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=24G
#SBATCH --mail-user=mingbinc@student.unimelb.edu.au
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END


source /data/projects/punim0478/bryan/anaconda3/bin/activate gpu_env
python3 quickstart_tutorial.py
sleep 60
python3 tensorqs_tutorial.py 

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s

