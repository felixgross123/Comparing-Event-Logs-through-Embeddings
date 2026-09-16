#!/usr/bin/zsh

### Job Parameters
#SBATCH --job-name=cwindow_timing   # Sets the job name
#SBATCH --output=stdout_%j.txt      # Redirects stdout and stderr, %j = job id
#SBATCH --time=03:00:00             # Max Runtime
#SBATCH --ntasks=1                  # One Python process, no MPI
#SBATCH --cpus-per-task=24          # CPU cores
#SBATCH --mem=16G                   # absolute memory
#SBATCH --partition=c23g            # CLAIX-2023 GPU partition (4x NVIDIA H100 per node)
#SBATCH --gres=gpu:1                # One GPU

### Program Code
set -euo pipefail

: ${SLURM_SUBMIT_DIR:=${0:A:h}}
cd "$SLURM_SUBMIT_DIR"

### Software environment
module load Python
module load CUDA

export PIP_CACHE_DIR="${HPCWORK:-$PWD}/.pip-cache"
export OMP_NUM_THREADS=${TORCH_THREADS:-4}          # torch threads
export RAYON_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}  # EMSC threads

if [[ ! -d logs ]]; then
    echo "ERROR: no logs/ folder in $PWD" >&2
    exit 1
fi

if [[ ! -d .venv ]]; then
    echo "== creating .venv =="
    python3 -m venv .venv
fi
source .venv/bin/activate

echo "== installing dependencies =="
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

### Job info
echo "== job ${SLURM_JOB_ID:-none} on $(hostname), $(date) =="
echo "== $(python --version) from $(which python), ${OMP_NUM_THREADS} CPU threads =="
if command -v nvidia-smi > /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
fi
python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"

### Benchmark
echo "== running the benchmark =="
python timing_benchmark.py

echo "== done: $PWD/timing_results.txt =="
