#!/bin/bash
#SBATCH --job-name=llm-predict
#SBATCH --account=torch_pr_932_general
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=4:00:00
#SBATCH --output=llm-predict.log
#SBATCH --error=llm-predict.log

set -euo pipefail

REPO_ROOT=${SCRATCH}/llm-key-estimation

IMAGE=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVERLAY=${REPO_ROOT}/conda-overlay.ext3:rw

CHORDS_CSV=${CHORDS_CSV:-chords/chords.csv}
OUTPUT=${OUTPUT:-method/llm-predictions-claude.csv}
PROVIDER=${PROVIDER:-claude}
LIMIT=${LIMIT:-500}

export CHORDS_CSV OUTPUT PROVIDER LIMIT

cd "${REPO_ROOT}"

echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node(s): ${SLURM_NODELIST:-unknown}"
echo "Working directory: $(pwd)"
echo "Chords CSV: ${CHORDS_CSV}"
echo "Output: ${OUTPUT}"
echo "Provider: ${PROVIDER}"
echo "Limit: ${LIMIT}"

singularity exec --fakeroot --overlay "${OVERLAY}" "${IMAGE}" /bin/bash -lc '
  cd "'"${REPO_ROOT}"'"
  source /ext3/env.sh
  python method/llm_predict.py \
    --chords-csv "${CHORDS_CSV}" \
    --output "${OUTPUT}" \
    --provider "${PROVIDER}" \
    --limit "${LIMIT}"
'
