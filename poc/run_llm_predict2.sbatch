#!/bin/bash
#SBATCH --job-name=llm-predict2-poc
#SBATCH --account=torch_pr_932_general
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=4:00:00
#SBATCH --output=llm-predict2-poc.log
#SBATCH --error=llm-predict2-poc.log

set -euo pipefail

REPO_ROOT=${SCRATCH}/llm-key-estimation

IMAGE=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVERLAY=${REPO_ROOT}/conda-overlay.ext3:rw

POC_CSV=${POC_CSV:-poc/poc.csv}
OUTPUT=${OUTPUT:-poc/llm-predictions2.csv}

export POC_CSV OUTPUT

cd "${REPO_ROOT}"

echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node(s): ${SLURM_NODELIST:-unknown}"
echo "Working directory: $(pwd)"
echo "POC CSV: ${POC_CSV}"
echo "Output: ${OUTPUT}"

singularity exec --fakeroot --overlay "${OVERLAY}" "${IMAGE}" /bin/bash -lc '
  cd "'"${REPO_ROOT}"'"
  source /ext3/env.sh
  python poc/llm_predict2.py \
    --poc-csv "${POC_CSV}" \
    --output "${OUTPUT}"
'
