#!/bin/bash
#SBATCH --job-name=allconv-predict
#SBATCH --account=torch_pr_932_general
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.log
#SBATCH --error=%x-%j.log

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

IMAGE=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVERLAY=conda-overlay.ext3:rw

DATA_DIR=${DATA_DIR:-fma-keys}
OUTPUT=${OUTPUT:-baseline/allconv-predictions.csv}
NUM_WORKERS=${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-16}}

export DATA_DIR OUTPUT NUM_WORKERS

cd "${REPO_ROOT}"

echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node(s): ${SLURM_NODELIST:-unknown}"
echo "Working directory: $(pwd)"
echo "Data dir: ${DATA_DIR}"
echo "Output: ${OUTPUT}"
echo "Workers: ${NUM_WORKERS}"

singularity exec --fakeroot --nv --overlay "${OVERLAY}" "${IMAGE}" /bin/bash -lc '
  cd "'"${REPO_ROOT}"'"
  source /ext3/env.sh
  python baseline/allconv_predict.py \
    --data-dir "${DATA_DIR}" \
    --output "${OUTPUT}" \
    --num-workers "${NUM_WORKERS}"
'