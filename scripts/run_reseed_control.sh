#!/bin/zsh
# Week 3 control: the same 2,000 episodes at full precision with only the planner's random seed
# changed. Shows how often a different search alone changes the plan, the noise floor for
# quantization-induced behaviour changes. Waits for the quantization sweep to finish first.
# Usage: scripts/run_reseed_control.sh
cd "${0:A:h}/.."
name=fp32_reseed
offset=100000

while pgrep -f "run_quantization_sweep.sh" >/dev/null; do sleep 60; done
if ! grep -q "^1000/1000" data/$name/log_0.txt 2>/dev/null || ! grep -q "^1000/1000" data/$name/log_1000.txt 2>/dev/null; then
  echo "$(date '+%F %T') collecting $name"
  mkdir -p data/$name
  for f in 0 1000; do
    .venv/bin/python -W ignore scripts/collect.py --name $name --precision fp32 --cem-seed-offset $offset \
      --first $f --episodes 1000 --threads 4 > data/$name/log_$f.txt 2>&1 &
  done
  wait
fi
echo "$(date '+%F %T') validating $name"
.venv/bin/python -W ignore scripts/validate_data.py data/$name/episodes_*.npz --rerun 2 > data/$name/validation.txt 2>&1
tail -n 1 data/$name/validation.txt
echo "$(date '+%F %T') control done"
