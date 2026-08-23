#!/bin/bash
#SBATCH --job-name=classify-abstracts
#SBATCH --account=torch_pr_932_general
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=1:00:00
#SBATCH --output=classify-abstracts.log
#SBATCH --error=classify-abstracts.log

set -euo pipefail

REPO_ROOT=${SCRATCH}/llm-key-estimation

IMAGE=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVERLAY=${REPO_ROOT}/conda-overlay.ext3:rw

ABSTRACTS_CSV=${ABSTRACTS_CSV:-ismir-papers/abstracts.csv}
OUTPUT=${OUTPUT:-ismir-papers/classifications.csv}

export ABSTRACTS_CSV OUTPUT

cd "${REPO_ROOT}"

echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node(s): ${SLURM_NODELIST:-unknown}"
echo "Working directory: $(pwd)"
echo "Abstracts CSV: ${ABSTRACTS_CSV}"
echo "Output: ${OUTPUT}"

singularity exec --fakeroot --overlay "${OVERLAY}" "${IMAGE}" /bin/bash -lc '
  cd "'"${REPO_ROOT}"'"
  source /ext3/env.sh
  python ismir-papers/classify_abstracts.py \
    --abstracts-csv "${ABSTRACTS_CSV}" \
    --output "${OUTPUT}"
'
