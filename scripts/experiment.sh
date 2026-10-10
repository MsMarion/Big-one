#!/usr/bin/env bash
# New data -> train -> test benchmark -> report, in one command.
#   bash scripts/experiment.sh experiments/<name>.yaml [--data <local_dir>] [--force]
# --data uploads <local_dir> to the experiment's data.root on the box first.
# Runs detached; follow with the printed tail command, the dashboard, or the report in JupyterLab.
set -euo pipefail
. "$(dirname "$0")/_common.sh"
EXP=$1; shift
DATA=""; EXTRA=""
while [ $# -gt 0 ]; do
  case $1 in
    --data) DATA=$2; shift 2 ;;
    --force) EXTRA="--force"; shift ;;
    *) echo "unknown arg $1"; exit 1 ;;
  esac
done
[ -f "$REPO/$EXP" ] || [ -f "$EXP" ] || { echo "no such experiment file: $EXP"; exit 1; }
EXP_REL=$(cd "$REPO" && realpath --relative-to=. "$EXP")

bash "$(dirname "$0")/sync.sh"
read -r NAME ROOT < <(ssh "$HOST" "source $REMOTE_CODE/remote/env.sh && cd $REMOTE_CODE && python -c \"
from dino.experiment import load_experiment as L; e = L('$EXP_REL'); print(e['name'], e['data']['root'])\"")
echo "experiment: $NAME   data root: $ROOT"

if [ -n "$DATA" ]; then
  echo "uploading $DATA -> $HOST:$ROOT ..."
  ssh "$HOST" "mkdir -p $ROOT"
  tar czf - -C "$DATA" . | ssh "$HOST" "tar xzf - --no-same-owner -C $ROOT"
fi

bash "$(dirname "$0")/bg.sh" "exp_$NAME" python -m dino.experiment "$EXP_REL" $EXTRA
cat <<MSG

Running. Follow it:
  bash scripts/run.sh tail -f /workspace/dino/runs/_jobs/exp_$NAME.log
Report (updates after every run):  JupyterLab -> dino/reports/$NAME.html
Local copy:  bash scripts/pull_report.sh $NAME
MSG
