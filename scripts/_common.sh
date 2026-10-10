# Shared settings for local helper scripts (run from Git Bash).
HOST=${DINO_HOST:-dino-box}
REMOTE_ROOT=/workspace/dino
REMOTE_CODE=$REMOTE_ROOT/code
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
