#!/bin/bash
# One segment of a chained AANCA training run. Submit it through
# ``scripts/bunya_submit.sh``, which sets the resources; every argument here is
# passed straight to ``scripts/train.py``.
#
#   (what the submitter does) sbatch <resources> bunya_train.slurm --tag a --updates 100000
#
# The segment does three things: run the training until it either finishes or
# runs out of its time budget, write a log either way, and resubmit itself if
# and only if there is more to do.
#
# **The device, the thread count and the time budget are derived here, from the
# allocation this job actually got**, rather than being passed in. That way a
# resubmission needs nothing but the scientific arguments, and a segment that
# lands on a different partition configures itself correctly instead of
# inheriting the previous segment's assumptions.
#
# **Exit codes carry the decision.** 0 means finished, or stopped on purpose by
# the stop file. 64 means "checkpointed, not finished" -- the only code that
# resubmits. Anything else is a crash, and a crash must not be retried in a
# loop; it stops the chain and leaves the log for a human.
#SBATCH --job-name=aanca
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -uo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${ROOT}" || exit 1
mkdir -p logs

PROFILE="${AANCA_PROFILE:-cpu}"
CHAIN="${AANCA_CHAIN:-1}"
MARGIN="${AANCA_MARGIN:-900}"
THREADS="${SLURM_CPUS_PER_TASK:-4}"

# Time budget: stop cleanly this long before SLURM would kill the job. The
# USR1 signal at 300 s is the primary mechanism and this is the belt to its
# braces -- it also covers a job whose signal is missed or arrives late.
NOW="$(date +%s)"
if [[ -n "${SLURM_JOB_END_TIME:-}" ]]; then
    BUDGET=$(( SLURM_JOB_END_TIME - NOW - MARGIN ))
else
    BUDGET=$(( ${AANCA_WALL_SECONDS:-86400} - MARGIN ))
fi
(( BUDGET < 300 )) && BUDGET=300

DEVICE=cpu
[[ "${PROFILE}" == "gpu" || -n "${SLURM_JOB_GPUS:-}${SLURM_GPUS_ON_NODE:-}" ]] && DEVICE=cuda

echo "job        : ${SLURM_JOB_ID:-none} on $(hostname)"
echo "segment    : ${CHAIN} (profile ${PROFILE})"
echo "root       : ${ROOT}"
echo "device     : ${DEVICE}, ${THREADS} threads"
echo "budget     : ${BUDGET} s (margin ${MARGIN} s)"
echo "args       : $*"
echo "started    : $(date -Is)"

# The environment is built once on a login node by scripts/bunya_stage.sh. A
# job that builds its own environment spends its allocation on pip.
VENV="${AANCA_VENV:-${ROOT}/.venv_bunya}"
if [[ ! -x "${VENV}/bin/python" ]]; then
    echo "ERROR: no environment at ${VENV} -- run: bash scripts/bunya_stage.sh" >&2
    exit 1
fi
# The venv runs on a module interpreter, so load the module it was built
# against before activating; stage.sh recorded which one.
if [[ -r "${VENV}/module.txt" ]]; then
    # ``module`` is a shell function, and a non-login batch shell may not have
    # it. Source the initialisation rather than assume.
    if ! command -v module >/dev/null 2>&1; then
        for init in /etc/profile.d/lmod.sh /etc/profile.d/modules.sh                     "${LMOD_PKG:-/usr/share/lmod/lmod}/init/bash"; do
            [[ -r "${init}" ]] && source "${init}" && break
        done
    fi
    module purge 2>/dev/null || true
    module load "$(cat "${VENV}/module.txt")" 2>/dev/null || true
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
# Explicit thread counts: an oversubscribed node slows the neighbour search
# instead of speeding it up, and the count is also part of what makes a resume
# bit-identical (multithreaded reductions sum in thread-completion order).
export OMP_NUM_THREADS="${THREADS}"
export MKL_NUM_THREADS="${THREADS}"

python -c "
import torch
print('torch      :', torch.__version__)
print('hip / cuda :', getattr(torch.version, 'hip', None), getattr(torch.version, 'cuda', None))
print('gpu        :', torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
" || { echo "ERROR: torch does not import in ${VENV}" >&2; exit 1; }

# Run it as a child of this shell rather than under ``srun``, and forward the
# signals. ``--signal=B:USR1@300`` delivers to the *batch* shell, so a step
# launched with srun would never see the warning and would be killed outright
# at the walltime with its last interval unsaved. The trap turns SLURM's
# warning into train.py's clean stop.
python scripts/train.py "$@" --device "${DEVICE}" --threads "${THREADS}" \
    --max-seconds "${BUDGET}" &
CHILD=$!
forward() { kill -"$1" "${CHILD}" 2>/dev/null || true; }
trap 'echo "forwarding USR1"; forward USR1' USR1
trap 'echo "forwarding TERM"; forward TERM' TERM
trap 'echo "forwarding INT";  forward INT'  INT

# ``wait`` returns as soon as a trap runs, so keep waiting while the child is
# alive; otherwise the script would race ahead and report the trap's own exit
# status as the run's.
while true; do
    wait "${CHILD}"
    RC=$?
    kill -0 "${CHILD}" 2>/dev/null || break
done

echo "train exit : ${RC}"
echo "finished   : $(date -Is)"

# --- the chain --------------------------------------------------------------
# 64 is train.py's "checkpointed, not finished". The stop file is checked again
# here because it may have been created after the last update was examined, and
# a resubmission after a deliberate stop is the one failure that would be hard
# to notice: the run would simply carry on.
if (( RC == 64 )); then
    TAG="run"
    args=("$@")
    for ((i = 0; i < ${#args[@]}; i++)); do
        [[ "${args[i]}" == "--tag" ]] && TAG="${args[i + 1]}"
    done
    if [[ -e "${ROOT}/runs/${TAG}/STOP" ]]; then
        echo "stop file present for tag ${TAG}; chain ends here"
        exit 0
    fi
    echo "resubmitting segment $((CHAIN + 1))"
    AANCA_CHAIN=$((CHAIN + 1)) bash "${ROOT}/scripts/bunya_submit.sh" "${PROFILE}" "$@"
    exit 0
fi

exit "${RC}"
