#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --mem=12GB
#SBATCH --job-name=ExtremeWeather
#SBATCH --mail-type=END
#SBATCH --mail-user=rtw262@nyu.edu
#SBATCH --output=slurm_%j.out
#SBATCH --gres=gpu:1

jupyter execute main.ipynb

