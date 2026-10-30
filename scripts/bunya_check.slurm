#!/bin/bash
# One card, long enough to ask whether the two neighbour selections agree
# across the training distribution, which is what switching a running
# experiment onto the new one requires. The risk is not a logic error -- the
# tests cover that -- but a residue pair within a float ulp of the cutoff,
# where cKDTree on the host and cdist on the card could disagree. Same
# resources as a training segment, so the arithmetic is the arithmetic the run
# would use.
#
#   sbatch scripts/bunya_check.slurm
#SBATCH --job-name=aanca-check
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

echo "host : $(hostname)"
python scripts/check_neighbours.py --structures 60 --perturbations 4
RC=$?
echo "exit : ${RC}"
exit "${RC}"
