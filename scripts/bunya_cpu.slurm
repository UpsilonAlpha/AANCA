#!/bin/bash
# A diagnostic that does not need a card, run where the queue is short.
#
#   sbatch scripts/bunya_cpu.slurm scripts/chi1_histogram.py \
#       runs/v5_native/best.pt --casp 13 --steps 100 --device cpu
#
# The GPU QOS has a per-user job limit shared with everything else Ulban
# runs, so a small diagnostic can sit behind a self-refilling array for
# hours. The analysis scripts are minutes-to-an-hour of CPU, which the
# ``general`` partition will schedule immediately.
#
# This exists because the first two attempts were run in the code-server
# terminal instead, and both died: once when the OnDemand session moved node,
# taking its /tmp with it. A batch job survives that.
#SBATCH --job-name=aanca-cpu
#SBATCH --partition=general
#SBATCH --account=a_imb_ccbcd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${ROOT}" || exit 1
mkdir -p logs

if [[ $# -lt 1 ]]; then
    echo "usage: sbatch scripts/bunya_cpu.slurm <script.py> [args...]"
    exit 2
fi

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
echo "run  : $*"
echo
python "$@"
RC=$?
echo
echo "exit : ${RC}"
exit "${RC}"
