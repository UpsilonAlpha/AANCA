#!/bin/bash
# A diagnostic that does not need a card, run where the queue is short.
#
#   sbatch --constraint=$(cat .venv_bunya/arch.txt) scripts/bunya_cpu.slurm \
#       scripts/chi1_histogram.py runs/v5_native/best.pt --casp 13 --steps 100
#
# **The --constraint is not optional, and leaving it off is a coin flip.** The
# venv is built for one CPU generation (epyc5); on an epyc3 node the job dies
# in about two seconds with SIGILL -- exit 132, an empty .out and "Illegal
# instruction" -- before running any project code. A #SBATCH line cannot read
# arch.txt, so it has to be passed at submit time, and `bunya_submit.sh gpu`
# does it for you while a bare sbatch of this script does not. Paid for on
# 2026-11-02: steric_preflight.py landed on bun030 and died instantly, while
# two clamp jobs submitted the same way the same hour happened to land on
# bun151 and ran fine. Earlier CPU diagnostics may have died this way without
# anyone noticing, since the symptom is an empty log.
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
# -u, always. Every diagnostic on the night of 1-2 Nov wrote its whole output
# in one buffered flush at exit, which cost two things. There was no way to see
# how far a job had got -- the .out sat at these two header lines for hours, so
# "still running" and "hung" were indistinguishable -- and a job killed on its
# walltime would have left nothing at all, throwing away every target it had
# already packed. `bestof` came within 25 minutes of exactly that after five
# and three quarter hours. Unbuffered costs nothing here: these scripts print
# a few dozen lines, not a stream.
python -u "$@"
RC=$?
echo
echo "exit : ${RC}"
exit "${RC}"
