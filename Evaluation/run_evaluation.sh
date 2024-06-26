#!/bin/bash

#SBATCH --job-name=Try1.0      	      # Job name
#SBATCH --output=result.%j.out        # Standard output and error log (%j expands to jobId)
#SBATCH --error=result.%j.err
#SBATCH --ntasks=1                    # Number of tasks (processes)
#SBATCH --cpus-per-task=80            # Number of CPU cores per task
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --time=15:00:00               # Time limit hrs:min:sec
#SBATCH --mem=180000mb                    # Memory per node (optional)
#SBATCH --partition=single            # Partition

module load compiler/intel/2023.1.0
module load devel/python/3.12.3_intel_2023.1.0


# Run the python code
python load_evaluation.py 
