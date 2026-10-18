#!/bin/zsh
# Week 3: collect the 2,000 seeded episodes at each quantized precision, one precision at a time,
# two shards in parallel, then validate each. Precisions already collected are skipped, so the
# script can be rerun after an interruption. Usage: scripts/run_quantization_sweep.sh [precisions...]
cd "${0:A:h}/.."
precisions=(${@:-int8 w6 w4 w3})

for q in $precisions; do
  if [[ -f data/$q/episodes_0000-0999.npz && -f data/$q/episodes_1000-1999.npz ]] && \
     grep -q "^1000/1000" data/$q/log_0.txt data/$q/log_1000.txt 2>/dev/null && \
     [[ $(grep -c "^1000/1000" data/$q/log_0.txt data/$q/log_1000.txt | awk -F: '{s+=$2} END {print s}') -eq 2 ]]; then
    echo "$(date '+%F %T') $q already collected, skipping"
    continue
  fi
  # wait for any collection already running for this precision (e.g. started by hand)
  while pgrep -f "scripts/collect.py --name $q " >/dev/null; do sleep 60; done
  if ! grep -q "^1000/1000" data/$q/log_0.txt 2>/dev/null || ! grep -q "^1000/1000" data/$q/log_1000.txt 2>/dev/null; then
    echo "$(date '+%F %T') collecting $q"
    mkdir -p data/$q
    for f in 0 1000; do
      .venv/bin/python -W ignore scripts/collect.py --name $q --precision $q --first $f --episodes 1000 --threads 4 \
        > data/$q/log_$f.txt 2>&1 &
    done
    wait
  fi
  echo "$(date '+%F %T') validating $q"
  .venv/bin/python -W ignore scripts/validate_data.py data/$q/episodes_*.npz --rerun 2 > data/$q/validation.txt 2>&1
  tail -n 1 data/$q/validation.txt
done
echo "$(date '+%F %T') sweep done"
