#!/bin/bash
#SBATCH --partition=Dance        # Partition to run the job on
#SBATCH --job-name=hunyuan3d     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=20       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --time=00:10:00          # total run time limit (HH:MM:SS)
#SBATCH --gres=gpu:1             # number of gpus per node

# Print job details for debugging
echo "Job started at $(date)"
echo "Running on $(hostname)"

# Check if we received any command-line arguments
if [ $# -eq 0 ]; then
    echo "No arguments provided. Usage: sbatch $0 [command] [args...]"
    echo "Example: sbatch $0 text-to-3d --prompt \"a red sports car\""
    echo "Example: sbatch $0 fast --image /path/to/image.jpg"
    exit 1
fi

# Log the command being executed
echo "Executing: apptainer exec --nv image.sif $@"

# Execute apptainer with all the arguments passed to this script
apptainer run --nv image.sif "$@"

# Print job completion information
echo "Job completed at $(date)"
