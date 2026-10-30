#!/bin/bash
# Twenty minutes on one card to find out where a rollout step's time goes.
#
#   sbatch scripts/bunya_profile.slurm
#
# Separate from ``bunya_gpu_check.slurm`` because that one asks whether the
# wheel runs at all; this one asks why an update costs 2.8 s when the
# arithmetic in it is microseconds. Same resources as a training segment, so
# the numbers transfer: 8 CPUs and one MI355X.
#SBATCH --job-name=aanca-profile
#SBATCH --partition=gpu_rocm
#SBATCH --qos=gpu
#SBATCH --gres=gpu:mi355x:1
#SBATCH --account=a_imb_ccbcd
#SBATCH --constraint=epyc5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:20:00
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
python scripts/profile_step.py --crop-size 384 --steps 80
RC=$?
echo "exit : ${RC}"
exit "${RC}"
