#!/bin/zsh
# Week 4 (RQ4): run every steering condition, one process at a time (the laptop charger cannot
# keep up with two). Each condition resumes from its last checkpoint, so the script can be rerun.
# Usage: scripts/run_steering.sh            (the original 120 unsafe and 60 random episodes)
#        N_UNSAFE=240 N_RANDOM=240 scripts/run_steering.sh   (extends both sets; finished episodes are reused)
cd "${0:A:h}/.."
n_unsafe=${N_UNSAFE:-120}
n_random=${N_RANDOM:-60}
unsafe=("probe --lam 100" "random --lam 100" "geometry --lam 100" "probe --lam 30" "probe --lam 300" "random --lam 300"
        "geometry --lam 300" "steer --alpha 3" "steer --alpha 10" "steer_random --alpha 10")
random=("probe --lam 100" "probe --lam 300" "random --lam 100" "geometry --lam 100" "steer --alpha 10")
for spec in $random; do
  echo "$(date '+%F %T') random: $spec"
  .venv/bin/python -W ignore scripts/steering.py --mode ${=spec} --set random --episodes $n_random 2>&1 | grep -E "^$n_random/$n_random|resuming|rror|Trace|usage"
done
for spec in $unsafe; do
  echo "$(date '+%F %T') unsafe: $spec"
  .venv/bin/python -W ignore scripts/steering.py --mode ${=spec} --set unsafe --episodes $n_unsafe 2>&1 | grep -E "^$n_unsafe/$n_unsafe|resuming|rror|Trace|usage"
done
echo "$(date '+%F %T') steering done"
