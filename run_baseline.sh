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
#SBATCH --output=results/logs/slurm_%x_%j.out
#SBATCH --time=12:00:00

# usage: sbatch --job-name=<tag>_<mode> run_baseline.sh <mode> [seed] [tag]
# The tag names the *configuration*; without it two different gossip settings
# at the same mode+seed overwrite each other's results.

source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate objdet310
cd ~/OBJ_DET
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Preflight: this workload needs ~13 GB of a 16 GB P100, so it only fits on a
# GPU nobody else is on. SLURM's gres accounting does not see processes
# started outside SLURM, and job 37452 died after two minutes because the
# card it was handed already had 3.15 GB in use by another user. Fail in
# seconds with a readable reason instead of an OOM traceback later.
FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
echo "preflight: $(hostname) GPU free = ${FREE_MIB} MiB"
if [ "${FREE_MIB:-0}" -lt 15000 ]; then
    echo "ABORT: only ${FREE_MIB} MiB free - another process is on this GPU."
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv
    exit 1
fi

python -u train.py --mode "$1" --seed "${2:-42}" --tag "${3:-}"
