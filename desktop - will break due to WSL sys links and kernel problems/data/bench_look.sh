#!/usr/bin/env bash
source /opt/ros/jazzy/setup.bash
source /ws/install/setup.bash
python3 /repo/tools/run_benchmark.py "$@" > /tmp/bench_out.txt 2>&1
tail -4 /tmp/bench_out.txt | head -1
grep -E "^\[ *[0-9]+/" /tmp/bench_out.txt
d=$(ls -td /data/runs/*_benchmark_* | head -1)
python3 -c "import json,sys; [print((json.loads(l).get(\"look\") or \"-\")[:170]) for l in open(sys.argv[1])]" $d/episodes.jsonl
grep -E "success_rate|collisions\"|RESULT" /tmp/bench_out.txt
