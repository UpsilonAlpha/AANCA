#!/bin/bash
# Reproduce the non-finite failure under instrumentation, on a CPU node.
#
#   sbatch --constraint=$(cat .venv_bunya/arch.txt) scripts/bunya_nan.slurm
#   sbatch --constraint=epyc5 scripts/bunya_nan.slurm --scan
#
# Everything after the script name goes to ``scripts/find_nan.py``.
#
# **Here rather than on a workstation**, because the failure took about 500
# updates to appear and the cluster does an update in a couple of seconds
# against half a minute on a laptop -- the difference between an answer in
# twenty minutes and an answer tomorrow. This is also the machine the failure
# actually happened on, so a reproduction here is a reproduction rather than
# an analogy.
#
# Short and CPU-only on purpose: it must not sit in the GPU queue behind the
# training chains it exists to unblock.
#SBATCH --job-name=aanca-nan
#SBATCH --partition=general
#SBATCH --account=a_imb_ccbcd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${ROOT}" || exit 1
mkdir -p logs

VENV="${AANCA_VENV:-${ROOT}/.venv_bunya}"
if [[ -r "${VENV}/arch.txt" ]]; then
    WANT="$(cat "${VENV}/arch.txt")"
    HAVE="$(sinfo -h -n "$(hostname -s)" -o '%f' 2>/dev/null | tr ',' '\n' | grep -m1 '^epyc' || true)"
    if [[ -n "${WANT}" && -n "${HAVE}" && "${WANT}" != "${HAVE}" ]]; then
        echo "ERROR: ${VENV} was built for ${WANT}, this node is ${HAVE}." >&2
        echo "       Submit with --constraint=${WANT}." >&2
        exit 1
    fi
fi
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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

echo "host    : $(hostname)"
echo "started : $(date -Is)"
echo "args    : $*"

# -u because the useful output is the trajectory, and a trajectory that
# arrives only at exit is no use while the job is running.
python -u scripts/find_nan.py "$@"
RC=$?
echo "exit    : ${RC}  $(date -Is)"
exit "${RC}"
