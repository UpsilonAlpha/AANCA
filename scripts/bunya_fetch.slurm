#!/bin/bash
# Download the coordinates named in the manifest, on a CPU node.
#
#   sbatch scripts/bunya_fetch.slurm --split train
#   sbatch scripts/bunya_fetch.slurm --split validation
#
# **Cancellable and resumable by construction**, with no checkpoint to keep:
# ``scripts/fetch_structures.py`` lists what is already on disk and fetches only
# the rest, so killing it costs the entry in flight and nothing else. Resuming
# is re-running the same command.
#
# On a CPU node rather than a login node because the training split is ~24k
# entries at roughly half a megabyte each -- a few hours of serial HTTP, which
# is exactly the kind of thing a login node should not be doing.
#SBATCH --job-name=aanca-fetch
#SBATCH --partition=general
#SBATCH --account=a_imb_ccbcd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -uo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${ROOT}" || exit 1
mkdir -p logs

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

echo "host    : $(hostname)"
echo "started : $(date -Is)"
echo "args    : $*"

# Check the inode quota first: this writes one small file per entry, and the
# file count on this account binds long before the bytes do. A quota failure
# mid-download looks like a network problem and is not one.
rquota 2>/dev/null || echo "(rquota unavailable here)"

python scripts/fetch_structures.py "$@"
RC=$?
echo "exit    : ${RC}"
echo "on disk : $(find data/pdb -name '*.bcif' | wc -l) entries"
echo "finished: $(date -Is)"
exit "${RC}"
