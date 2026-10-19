#!/bin/zsh
# Week 4 (RQ4): run every steering condition, one process at a time (the laptop charger cannot
# keep up with two). Each condition resumes from its last checkpoint, so the script can be rerun.
# Usage: scripts/run_steering.sh
cd "${0:A:h}/.."
unsafe=("probe --lam 100" "random --lam 100" "geometry --lam 100" "probe --lam 30" "probe --lam 300" "random --lam 300"
        "geometry --lam 300" "steer --alpha 3" "steer --alpha 10" "steer_random --alpha 10")
random=("probe --lam 100" "probe --lam 300" "random --lam 100" "geometry --lam 100" "steer --alpha 10")
for spec in $unsafe; do
  echo "$(date '+%F %T') unsafe: $spec"
  .venv/bin/python -W ignore scripts/steering.py --mode ${=spec} --set unsafe --episodes 120 2>&1 | grep -E "^120/120|rror|Trace|usage"
done
for spec in $random; do
  echo "$(date '+%F %T') random: $spec"
  .venv/bin/python -W ignore scripts/steering.py --mode ${=spec} --set random --episodes 60 2>&1 | grep -E "^60/60|rror|Trace|usage"
done
echo "$(date '+%F %T') steering done"
