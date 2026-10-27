#!/bin/bash
# Run with:  source ~/esm2_v3/resume.sh
cd ~/esm2_v3
source ~/miniconda3/etc/profile.d/conda.sh && conda activate esmprobe
echo "=== queue ==="; squeue -u $USER
echo; echo "=== results_v2 files per tag (full: 42 each) ==="
for d in results_v2/probe_*; do echo "$d $(ls $d | wc -l)"; done
echo; echo "=== frozen data ==="
sha256sum -c data/CHECKSUMS.txt 2>&1 | grep -v ': OK' || echo "frozen data OK"
echo; echo "=== what to do ==="
if squeue -u $USER | grep -q -E "pv2|probe"; then echo "jobs still running or queued: wait, then re-run."
else echo "no jobs in queue: if any tag has fewer than 42 files, run: sbatch probe_v2_a.slurm and/or sbatch probe_v2_b.slurm (finished files are skipped)"; fi
