#!/bin/bash

# Example of running python script in a batch mode

#SBATCH -J dataset_preprocess
#SBATCH -p normal
#SBATCH -c 1                            # one CPU core
#SBATCH -o hello-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=k.smirnov@innopolis.university

# Load software
module load anaconda3

# Run python script
srun python clevr_with_masks.py