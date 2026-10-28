#!/bin/bash
# One-time environment build for AANCA on Bunya, on a LOGIN node.
#
#   bash scripts/bunya_stage.sh            # CPU wheels (works everywhere)
#   AANCA_ROCM=7.0 bash scripts/bunya_stage.sh   # ROCm wheels for gpu_rocm
#
# Kept out of the batch job on purpose: a dependency resolution failure should
# cost a login shell, not a GPU allocation, and the job script stays a pure
# "run the thing".
#
# **Which wheel: 7.0, and this is not a preference.** A ROCm wheel only runs on
# the architectures it was compiled for, and ``rocm6.4`` has no gfx950 -- which
# is the MI355X, the only ROCm card on this cluster the environment can
# actually reach. Measured on bun161, 29 Oct: rocminfo prints
# ``amdgcn-amd-amdhsa--gfx950``, and the same job's torch 2.9.1+rocm6.4 reports
# ``is_available() False`` and ``get_arch_list() []``, then raises "No HIP GPUs
# are available" on the first ``.to(device)``. Note what that failure is *not*:
# not a kernel error, not a wrong answer, but a card the runtime declines to
# enumerate at all, because HIP hides devices no compiled target matches. The
# account's own working MI355X environment (``/scratch/user/uquadhik/.venv_esm``)
# runs torch 2.10.0+rocm7.0, which is where the version below comes from.
#
# The GPU path installs and then *checks*, in this script, rather than
# discovering it inside a queued job -- though the check is weak on a login
# node, because ``get_arch_list()`` is empty without a GPU either way. If
# the check fails, the CPU profile is still a complete answer: this model is
# small (a few hundred thousand parameters over a few hundred residues), its
# rollout is sequential, and its neighbour search is NumPy on the host, so the
# CPU partition is a reasonable place to train it and does not consume the
# account's four GPU slots.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${AANCA_VENV:-${ROOT}/.venv_bunya}"
ROCM="${AANCA_ROCM:-}"

echo "root  : ${ROOT}"
echo "venv  : ${VENV}"
echo "wheels: ${ROCM:+ROCm }${ROCM:-CPU-only}"

module purge 2>/dev/null || true
PYTHON=""
for candidate in python/3.12.3-gcccore-13.3.0 python/3.11.5-gcccore-13.2.0; do
    if module load "${candidate}" 2>/dev/null; then PYTHON="${candidate}"; break; fi
done
if [[ -z "${PYTHON}" ]]; then
    # Fall back to whatever 3.11+ the module tree offers, newest first.
    while read -r candidate; do
        [[ -z "${candidate}" ]] && continue
        if module load "${candidate}" 2>/dev/null; then
            version="$(python3 -c 'import sys; print("%d%02d" % sys.version_info[:2])' 2>/dev/null || echo 0)"
            if (( version >= 311 )); then PYTHON="${candidate}"; break; fi
            module unload "${candidate}" 2>/dev/null || true
        fi
    done < <(module -t avail python 2>&1 | grep -E '^python/3\.(1[1-9])' | sort -rV || true)
fi
if [[ -z "${PYTHON}" ]]; then
    echo "ERROR: no Python 3.11+ module found; 'module avail python' to look" >&2
    exit 1
fi
echo "module: ${PYTHON} ($(python3 -V))"

if [[ ! -x "${VENV}/bin/python" ]]; then
    python3 -m venv "${VENV}"
fi
# A venv built on a module's interpreter needs that module loaded to run: the
# shared library it links against lives in the module tree, not in the venv.
# Record which one, so the batch scripts load the same one rather than
# guessing, and so a rebuild against a different module is visible.
echo "${PYTHON}" > "${VENV}/module.txt"
# And the microarchitecture. Bunya keeps a separate software tree per CPU
# generation (/sw/auto/rocky9a/epyc5/...), so a venv over a module interpreter
# is built for one of them: measured here, a job that landed on an epyc3 node
# died instantly with "Illegal instruction (core dumped)" before running a line
# of this project's code. Jobs are constrained to this feature, and a second
# architecture means a second venv (AANCA_VENV=.venv_bunya_epyc3).
ARCH="$(sinfo -h -n "$(hostname -s)" -o '%f' 2>/dev/null | tr ',' '\n' | grep -m1 '^epyc' || true)"
echo "${ARCH}" > "${VENV}/arch.txt"
echo "arch  : ${ARCH:-unknown}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
python -m pip install --quiet --upgrade pip wheel

# ``--upgrade`` is load-bearing when restaging: without it, pip sees a torch
# already installed, reports "requirement already satisfied" and leaves the
# old wheel in place -- so switching index from rocm6.4 to rocm7.0 looks like
# it worked and changes nothing. The symptom is a staging log that prints the
# version it was supposed to replace.
if [[ -n "${ROCM}" ]]; then
    python -m pip install --quiet --upgrade torch --index-url "https://download.pytorch.org/whl/rocm${ROCM}"
else
    python -m pip install --quiet --upgrade torch --index-url "https://download.pytorch.org/whl/cpu"
fi
python -m pip install --quiet "numpy>=2.0" "biotite>=1.0" "pytest>=8.0"
python -m pip install --quiet --no-deps -e "${ROOT}"

echo
python - <<'PY'
import torch
print("torch        :", torch.__version__)
print("hip / cuda   :", getattr(torch.version, "hip", None), getattr(torch.version, "cuda", None))
print("gpu visible  :", torch.cuda.is_available(), "(a login node has none; this is expected)")
# The architectures this wheel was compiled for. Compare against the card's
# gfx -- MI210 is gfx90a, MI300X gfx942, MI355X gfx950 -- because a wheel
# missing the target gfx fails at the first kernel launch inside the job, not
# here. ``rocminfo | grep gfx`` on an allocation prints the card's own name.
print("arch list    :", torch.cuda.get_arch_list())
PY

echo
echo "running the test suite (no GPU needed; proves the checkout is sound)"
python -m pytest -q -m "not slow" "${ROOT}/tests" | tail -3

cat <<'EOF'

Next:
  # the coordinates, on a CPU node (resumable: it skips what is already there)
  sbatch scripts/bunya_fetch.slurm --split train

  # a training chain
  bash scripts/bunya_submit.sh cpu --tag cpu_a --updates 100000 --dry-run
EOF
