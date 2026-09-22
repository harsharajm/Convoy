#!/bin/bash
#SBATCH --partition=cse-gpu-all
# Any of the four P100-PCIE-16GB nodes (cse-node[009-012]), not just 009.
# They are identical hardware and the data sits on shared /u/student, so
# pinning one node only serialised every job behind the others while three
# equivalent nodes sat eligible.
#
# Expressed as an exclusion, NOT --nodelist. SLURM's --nodelist means "give
# me all of these hosts": --nodelist=cse-node[009-012] asks for four nodes
# at once for a one-GPU job, which never schedules. Excluding the non-P100
# members of the partition leaves exactly the four we want, one of them.
#
# dgx-v100-01 is excluded on purpose even though its cards are larger
# (32GB): the differences being measured here are ~0.005 mAP, and splitting
# a comparison across two GPU generations adds a confound for no gain.
#SBATCH --exclude=dgx-a100-01,dgx-a100-02,dgx-p100-01,dgx-v100-01
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --output=results/logs/slurm_%x_%j.out
# No --time: cse-gpu-all's own TIMELIMIT is "infinite" (sinfo -s), so omitting
# this just inherits that. A fixed --time=12:00:00 lived here before and
# silently killed the epoch8 joint run (job 37900) at 94% complete, ~30-45min
# from finishing, with no partial results saved - EPOCHS_PER_TASK overrides
# and --eval-every-epoch both push runtime well past what was sized for the
# original 3-epoch baseline this script was written for. Self-imposed caps on
# variable-length experiments cost more than they protect against.

# usage: sbatch --job-name=<tag>_<mode> run_baseline.sh <mode> [seed] [tag] [extra train.py args...]
# The tag names the *configuration*; without it two different gossip settings
# at the same mode+seed overwrite each other's results.
#
# Anything after the third positional arg is forwarded verbatim to train.py -
# e.g. --epochs 8 --lr-decay --freeze-backbone-stages 1. This is what lets
# two back-to-back sbatch submissions use different joint configurations
# without touching config.py: sbatch returns as soon as a job is queued, not
# when it starts, so editing config.py for the second job before the first
# has actually started running would silently change the first job's setting
# too.

source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate objdet310

# cd to the directory the job was submitted from, not a hardcoded path. The
# v2 copy said `cd ~/OBJ_DET`, so submitting this from OBJ_DET/v3 would have
# run v2's train.py against v2's config and written v2 results under a v3
# tag - a silent wrong answer, not a crash.
#
# SLURM_SUBMIT_DIR, not $0: sbatch copies the script to
# /var/spool/slurmd/job<id>/ before running it, so $0 points at the spool
# copy and `dirname $0` lands in a directory with no source in it. The $0
# form only looks right when the script is invoked directly with bash, which
# is exactly how it was first (and wrongly) tested. The fallback keeps direct
# invocation working.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "$0")")}" || exit 1
echo "workdir: $(pwd)"
# Fail now if we are not where the code is, rather than 3 hours from now.
[ -f train.py ] && [ -f config.py ] || { echo "ABORT: no train.py here"; exit 1; }

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Preflight: this workload needs ~13 GB of a 16 GB P100, so it only fits on a
# GPU nobody else is on. SLURM's gres accounting does not see processes
# started outside SLURM, and job 37452 died after two minutes because the
# card it was handed already had 3.15 GB in use by another user. Fail in
# seconds with a readable reason instead of an OOM traceback later.
FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
echo "preflight: $(hostname) GPU free = ${FREE_MIB} MiB"
if [ "${FREE_MIB:-0}" -lt 15000 ]; then
    echo "ABORT: only ${FREE_MIB} MiB free on $(hostname) - another process is here."
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv
    # Requeue rather than fail. Which GPU we land on is luck - the P100s are
    # shared and SLURM's gres accounting cannot see processes started outside
    # it - so failing outright means hand-resubmitting until a clean card
    # comes up. Bounded, because a cluster-wide shortage should surface as a
    # dead job, not an infinite retry loop.
    ATTEMPT=${SLURM_RESTART_COUNT:-0}
    if [ "$ATTEMPT" -lt 12 ]; then
        echo "requeueing (attempt $((ATTEMPT + 1))/12)"
        scontrol requeue "$SLURM_JOB_ID"
        sleep 30
    else
        echo "giving up after $ATTEMPT retries - no clean GPU available"
    fi
    exit 1
fi

python -u train.py --mode "$1" --seed "${2:-42}" --tag "${3:-}" "${@:4}"
