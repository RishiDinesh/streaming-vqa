# AGENTS.md

## Environment

- Always use the Conda environment named duo to run any Python scripts. This environment includes all required dependencies, such as: numpy, matplotlib, torch, and other commonly used libraries
- For any tasks requiring GPU resources, use SLURM to submit jobs instead of running them locally. Check available GPU resources with `sinfo` and submit jobs using `sbatch` with appropriate resource requests.

## Tips:
- use conda run -n duo python -c "" to run python code.

## Files to ignore
- Ignore any files and folders inside untracked/ directory, as they are not relevant to the project and may contain temporary or old files that can cause confusion.

