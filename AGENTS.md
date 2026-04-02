# AGENTS.md

## Environment

- Always use the Conda environment named duo to run any Python scripts. This environment includes all required dependencies, such as: numpy, matplotlib, torch, and other commonly used libraries
- For any tasks requiring GPU resources, use SLURM to submit jobs instead of running them locally. Check available GPU resources with `sinfo` and submit jobs using `sbatch` with appropriate resource requests.