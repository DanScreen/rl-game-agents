#!/usr/bin/env bash
# Launch the standard training runs in the background, one log file per run.
#   ./scripts/train_all.sh                 # everything
#   ./scripts/train_all.sh cartpole_ppo    # just some configs (names from configs/)
# Follow progress with:  tail -f runs/<name>.log   or   tensorboard --logdir runs
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs
configs=("$@")
if [ ${#configs[@]} -eq 0 ]; then
  configs=(cartpole_dqn cartpole_ppo space_invaders_dqn space_invaders_ppo chess_td)
fi
for name in "${configs[@]}"; do
  nohup rlg train "configs/${name}.yaml" > "runs/${name}.log" 2>&1 &
  echo "started ${name} (pid $!) -> runs/${name}.log"
done
