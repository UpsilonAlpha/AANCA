#!/bin/bash
# Ten minutes on one GPU to answer "does this wheel run on this card at all".
#
#   sbatch --constraint=$(cat .venv_bunya/arch.txt) scripts/bunya_gpu_check.slurm
#
# Worth its own job because the failure it looks for is invisible until the
# card is in front of you. A ROCm wheel contains code for the architectures it
# was compiled for -- MI210 is gfx90a, MI300X gfx942, MI355X gfx950 -- and
# ``torch.cuda.get_arch_list()`` returns nothing on a node without a GPU, so
# the only way to find out is to ask on an allocation. Finding out here costs
# ten minutes; finding out inside a training chain costs the queue wait plus
# every segment that retried.
#
# What it caught, 29 Oct, bun161: torch 2.9.1+rocm6.4 on an MI355X reports
# ``is_available() False`` while rocminfo prints gfx950 two lines above. The
# mismatch does not announce itself as a mismatch -- HIP simply declines to
# enumerate a device no compiled target matches, so the message is the same
# one you would get on a node with no GPU at all. Hence the two lines below
# printing rocminfo's view and torch's view next to each other: the diagnosis
# is in the disagreement, and neither line alone contains it.
#SBATCH --job-name=aanca-gpucheck
#SBATCH --partition=gpu_rocm
#SBATCH --qos=gpu
#SBATCH --gres=gpu:mi355x:1
#SBATCH --account=a_imb_ccbcd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
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

echo "host : $(hostname)"
echo "card : $(rocm-smi --showproductname 2>/dev/null | grep -i 'card series' | head -2)"
rocminfo 2>/dev/null | grep -m2 'gfx' || echo "(no rocminfo)"

python - <<'PY'
import time
import torch

print("torch     :", torch.__version__)
print("hip       :", getattr(torch.version, "hip", None))
print("available :", torch.cuda.is_available())
print("arch list :", torch.cuda.get_arch_list())
if not torch.cuda.is_available():
    raise SystemExit(
        "no GPU visible to torch. If rocminfo above printed a gfx target, this "
        "is a wheel/architecture mismatch, not a missing card: reinstall with "
        "AANCA_ROCM=<version covering that target> bash scripts/bunya_stage.sh"
    )
print("device    :", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_properties(0))

# A kernel launch is the actual test: an architecture mismatch fails here, not
# at import.
a = torch.randn(4096, 4096, device="cuda")
b = a @ a
torch.cuda.synchronize()
print("matmul    :", float(b.sum()))

# And the shapes this project actually uses, which are small: a few hundred
# residues by 64 neighbours. The point of timing them is that a tiny kernel is
# dominated by launch overhead, so the GPU's advantage here is an open
# question rather than a given.
for n in (384, 1024):
    x = torch.randn(n, 64, 128, device="cuda")
    w = torch.randn(128, 128, device="cuda")
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(200):
        x = torch.relu(x @ w)
    torch.cuda.synchronize()
    print(f"200 x ({n}, 64, 128) @ (128, 128): {time.perf_counter() - start:.3f} s")
PY
RC=$?
echo "exit : ${RC}"
exit "${RC}"
