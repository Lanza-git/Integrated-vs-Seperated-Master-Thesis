#!/bin/bash

#SBATCH --job-name=Try1.0      	      		# Job name
#SBATCH --output=result.1.460650.%j.out        	# Standard output and error log (%j expands to jobId)
#SBATCH --error=result.1.460650.%j.err
#SBATCH --ntasks=1                    		# Number of tasks (processes)
#SBATCH --cpus-per-task=80            		# Number of CPU cores per task
#SBATCH --nodes=1                     		# Number of nodes
#SBATCH --time=5:00:00               		# Time limit hrs:min:sec
#SBATCH --mem=180000mb                    	# Memory per node (optional)
#SBATCH --partition=single            		# Partition

module load compiler/intel/2023.1.0
module load devel/python/3.12.3_intel_2023.1.0

# Define the dataset ID
DATASET_ID="set_460650"				# Define the ID of the Dataset that should be used
RISK_FACTOR=1					# Define the risk-factor that should be use (Default = 1)

# Run the python code
python IvsS_Main.py $DATASET_ID $RISK_FACTOR
