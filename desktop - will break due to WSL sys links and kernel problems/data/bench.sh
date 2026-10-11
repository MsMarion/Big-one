source /repo/setup/ros_env.sh >/dev/null 2>&1
cd /repo && timeout 900 python3 tools/run_benchmark.py "$@" 2>&1 | grep -E "^\[ *[0-9]+/|RESULT|success_rate"
