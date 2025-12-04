#!/bin/bash

#SBATCH --job-name=yahpo
#SBATCH --account=cmsc35370
#SBATCH --time=08:00:00
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=2000

cd /home/jshe/cmsc35370/llm_bbo
source venv/bin/activate

python yahpo_benchmark.py $1
