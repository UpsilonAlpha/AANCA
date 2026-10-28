#!/bin/bash
# Submit a resumable AANCA training run on Bunya, and chain it past the walltime.
#
#   bash scripts/bunya_submit.sh gpu --tag gpu_a --updates 100000
#   bash scripts/bunya_submit.sh cpu --tag cpu_a --updates 100000
#   bash scripts/bunya_submit.sh gpu --tag gpu_a --updates 100000 --dry-run
#
# Everything after the profile is passed through to ``scripts/train.py``, so the
# scientific settings live in the command line you typed and the resource
# settings live here. The job script derives the device, the thread count and
# the time budget from the allocation it gets, so a resubmission carries only
# the arguments below.
#
# **Why a chain rather than one long job.** Shorter jobs start sooner, a 48 h
# job that dies at 47 h costs 47 h, and the GPU QOS allows only four jobs at a
# time, so a chain of day-long segments both starts earlier and loses less.
# ``scripts/train.py`` checkpoints its entire state -- weights, optimiser,
# every RNG, the sample pool, the structure split -- so a segment boundary is
# invisible in the results: measured on this project, a run stopped and resumed
# is bit-identical to one that ran straight through at one thread.
#
# **Stopping it.** ``scancel`` stops the running segment, but to stop the chain
# create the stop file first, or the next segment is submitted:
#
#   touch runs/<tag>/STOP            # ends the chain at the next update
#   scancel <jobid>                  # optional: do not wait for that update
#
# Delete the file to resume later; the run picks up from its last checkpoint.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${1:-}"
shift || true

ACCOUNT="${AANCA_ACCOUNT:-a_imb_ccbcd}"
# This cluster has MI210 (gfx90a), MI300X (gfx942) and MI355X (gfx950) behind
# one ``gpu_rocm`` partition, and **the card is pinned on purpose**: a ROCm
# wheel carries code only for the architectures it was built for, so an
# untyped ``gpu:1`` makes the architecture a property of the queue. Pinning it
# turns a silent "No HIP GPUs are available" into a wait. mi355x because the
# environment is a rocm7.0 wheel; ``sinfo -p gpu_rocm -o '%n %G'`` prints what
# is actually there, and AANCA_GRES overrides this if the wheel changes.
GPU_PARTITION="${AANCA_GPU_PARTITION:-gpu_rocm}"
GPU_QOS="${AANCA_GPU_QOS:-gpu}"
GRES="${AANCA_GRES:-gpu:mi355x:1}"
CPU_PARTITION="${AANCA_CPU_PARTITION:-general}"
WALL="${AANCA_WALL:-24:00:00}"
CHAIN_LIMIT="${AANCA_CHAIN_LIMIT:-30}"
CHAIN="${AANCA_CHAIN:-1}"

DRY=0
ARGS=()
for arg in "$@"; do
    if [[ "${arg}" == "--dry-run" ]]; then DRY=1; else ARGS+=("${arg}"); fi
done

case "${PROFILE}" in
    gpu)
        # One card: the model is small and the rollout is sequential, so a
        # second GPU would sit idle. The CPU cores are for the neighbour search
        # and the structure parsing, which stay on the host.
        RESOURCES=(--partition="${GPU_PARTITION}" --qos="${GPU_QOS}"
                   --gres="${GRES}" --cpus-per-task=8 --mem=48G --time="${WALL}")
        ;;
    cpu)
        # Outside the GPU QOS, so a CPU segment runs *alongside* the four GPU
        # jobs the account is allowed rather than competing for one of them.
        RESOURCES=(--partition="${CPU_PARTITION}" --cpus-per-task=16 --mem=48G
                   --time="${WALL}")
        ;;
    *)
        echo "usage: bash scripts/bunya_submit.sh {gpu|cpu} [train.py args]" >&2
        exit 2
        ;;
esac

if (( CHAIN > CHAIN_LIMIT )); then
    echo "chain limit ${CHAIN_LIMIT} reached; not resubmitting" >&2
    exit 0
fi

# Seconds, for the job's fallback when SLURM_JOB_END_TIME is unset.
IFS=: read -r h m s <<< "${WALL}"
WALL_SECONDS=$(( 10#${h:-0} * 3600 + 10#${m:-0} * 60 + 10#${s:-0} ))

# Pin the CPU generation the environment was built for; see bunya_stage.sh.
# For the GPU profile this costs nothing -- the MI355X nodes are epyc5 anyway.
ARCH_FILE="${AANCA_VENV:-${ROOT}/.venv_bunya}/arch.txt"
CONSTRAINT=()
if [[ -s "${ARCH_FILE}" ]]; then
    CONSTRAINT=(--constraint="$(cat "${ARCH_FILE}")")
fi

# A one-off dependency for the first segment, e.g. "afterok:<fetch job>" so a
# chain can be queued before its data has finished downloading. Deliberately
# not inherited by the resubmission: bunya_train.slurm clears it, because by
# then the dependency has already been satisfied and a stale one would hold
# every later segment forever.
DEPENDENCY=()
if [[ -n "${AANCA_DEPENDENCY:-}" ]]; then
    DEPENDENCY=(--dependency="${AANCA_DEPENDENCY}")
fi

mkdir -p "${ROOT}/logs"
CMD=(sbatch --account="${ACCOUNT}" "${RESOURCES[@]}" "${CONSTRAINT[@]}" "${DEPENDENCY[@]}"
     --job-name="aanca-${PROFILE}" --signal=B:USR1@300
     --output="${ROOT}/logs/%x-%j.out" --error="${ROOT}/logs/%x-%j.err"
     --export="ALL,AANCA_PROFILE=${PROFILE},AANCA_CHAIN=${CHAIN},AANCA_WALL_SECONDS=${WALL_SECONDS}"
     "${ROOT}/scripts/bunya_train.slurm" "${ARGS[@]}")

echo "profile   : ${PROFILE} (segment ${CHAIN} of at most ${CHAIN_LIMIT})"
echo "train args: ${ARGS[*]}"
echo "sbatch    : ${CMD[*]}"
if (( DRY )); then
    echo "(dry run: add --test-only to the sbatch line to see the expected start)"
    exit 0
fi
"${CMD[@]}"
