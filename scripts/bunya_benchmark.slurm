#!/bin/bash
# Tier 1: score a trained rule on the published CASP tables, on a card.
#
#   sbatch scripts/bunya_benchmark.slurm runs/v5_native/best.pt 13
#   sbatch scripts/bunya_benchmark.slurm runs/v5_native/best.pt 14 200
#
# ``benchmark.py`` is minutes on an MI355X and hours on a CPU, because the
# CASP targets are whole proteins rather than the 96-residue crops training
# uses. ``validate_benchmark.py`` runs first in the same job, so the
# baselines' own numbers are re-derived beside ours every time a row is
# quoted -- a row is only comparable if the harness that produced it still
# recovers the published table.
#SBATCH --job-name=aanca-bench
#SBATCH --partition=gpu_rocm
#SBATCH --qos=gpu
#SBATCH --gres=gpu:mi355x:1
#SBATCH --account=a_imb_ccbcd
#SBATCH --constraint=epyc5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${ROOT}" || exit 1
mkdir -p logs

CHECKPOINT="${1:-runs/v5_native/best.pt}"
CASP="${2:-13}"
STEPS="${3:-100}"

VENV="${AANCA_VENV:-${ROOT}/.venv_bunya}"
if [[ -r "${VENV}/module.txt" ]]; then
    if ! command -v module >/dev/null 2>&1; then
        for init in /etc/profile.d/lmod.sh /etc/profile.d/modules.sh; do
            [[ -r "${init}" ]] && source "${init}" && break
        done
    fi
    module purge 2>/dev/null || true
    module load "$(cat "${VENV}/module.txt")" 2>/dev/null || true
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "host       : $(hostname)"
echo "checkpoint : ${CHECKPOINT}"
echo "casp       : ${CASP}, ${STEPS} steps"
echo

python scripts/validate_benchmark.py --casp "${CASP}"
echo
echo "==== BENCHMARK ===="
python scripts/benchmark.py "${CHECKPOINT}" --casp "${CASP}" --steps "${STEPS}"
RC=$?
echo "exit : ${RC}"
exit "${RC}"
