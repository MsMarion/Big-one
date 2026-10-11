#!/usr/bin/env bash
# Is the Windows<->container clock offset small enough for zenoh? (It rejects timestamps >500 ms in the
# future and logs "exceeding delta" errors for every message.) Run inside the pnp container:
#   bash ~/pnp_repo/docker/run_dev.sh "bash /repo/setup/zenoh_clock_check.sh [seconds]"
# Fix if it fails: see docs/learning/00_setup.md ("Clock sync").
WAIT="${1:-20}"
LOG=/tmp/zenoh.log
before=$(grep -c "exceeding delta" "$LOG")
sleep "$WAIT"
after=$(grep -c "exceeding delta" "$LOG")
new=$((after - before))
last=$(grep "exceeding delta" "$LOG" | tail -1 | grep -o "is rejected: [^)]*")
echo "zenoh timestamp errors in the last ${WAIT}s: $new  (total $after)"
[ -n "$last" ] && echo "most recent: $last"
if [ "$new" -eq 0 ]; then echo "RESULT: PASS (clocks agree within 500 ms)"; else echo "RESULT: FAIL"; exit 1; fi
