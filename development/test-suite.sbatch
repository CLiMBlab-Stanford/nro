#!/bin/bash
# Run the complete test suite in parallel on one Slurm node.
#
# Submit from a checkout that has a development environment (`./install --dev`):
#
#     sbatch --partition=PARTITION --account=ACCOUNT development/test-suite.sbatch
#     sbatch ... development/test-suite.sbatch tests/test_cli.py -x
#
# The job clones the checkout to node-local scratch, copies uncommitted changes
# over the clone, and runs both test tiers with one pytest-xdist worker per
# allocated CPU. Extra arguments go to pytest. The log is written to
# slurm-nro-tests-JOBID.out in the submission directory.
#
#SBATCH --job-name=nro-tests
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-nro-tests-%j.out

set -euo pipefail

checkout=$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)
python="$checkout/.nro-env/bin/python"
if [[ ! -x "$python" ]]; then
    echo "No development environment at $checkout/.nro-env; run ./install --dev" >&2
    exit 2
fi
workers="${SLURM_CPUS_PER_TASK:-$(nproc)}"

scratch=$(mktemp -d "${TMPDIR:-/tmp}/nro-tests.XXXXXX")
trap 'rm -rf "$scratch"' EXIT

# A shared clone keeps history and tags, which the release tests read.
git clone --quiet --shared --no-checkout "$checkout" "$scratch/repo"
git -C "$scratch/repo" checkout --quiet --detach "$(git -C "$checkout" rev-parse HEAD)"
# Copy the working tree's current files, including uncommitted and untracked
# ones that Git does not ignore, then remove tracked files deleted locally.
(
    cd "$checkout"
    git ls-files -z --cached --others --exclude-standard \
        | xargs -0 -r sh -c 'for path; do [ -e "$path" ] && printf "%s\0" "$path"; done' sh \
        | xargs -0 -r cp --parents -t "$scratch/repo"
    git ls-files -z --deleted | (cd "$scratch/repo" && xargs -0 -r rm -f --)
)

echo "Testing $(git -C "$checkout" rev-parse --short HEAD) from $checkout"
echo "Workers: $workers on $(hostname)"
cd "$scratch/repo"
export PYTHONPATH="$scratch/repo"
# One thread per worker: parallel workers would otherwise each start one
# OpenMP or BLAS thread per core, which oversubscribes the node and makes
# floating-point reductions depend on thread scheduling.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1
"$python" -m pytest -q -p no:cacheprovider -n "$workers" \
    -m "integration or not integration" "$@" tests
