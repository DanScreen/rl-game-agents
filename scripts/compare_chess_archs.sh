#!/usr/bin/env bash
# Train several network architectures with the *same* AlphaZero self-play procedure
# (configs/chess_az.yaml, same seed, same number of games), one process per
# architecture in parallel, then play them all against each other and the baselines.
#
#   ./scripts/compare_chess_archs.sh                       # defaults below
#   ./scripts/compare_chess_archs.sh total_games=500       # extra overrides for every run
#
# Results: runs/chess_az_<arch>-*/ (curves, checkpoints, summary.json with parameter
# counts and wall-clock), and benchmarks/chess_arch_comparison.md (Elo table).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs benchmarks

declare -A MODELS=(
  [mlp]="{arch: mlp, hidden: [512, 512]}"
  [cnn]="{arch: cnn, layers: 6, channels: 64}"
  [resnet]="{arch: resnet, blocks: 3, channels: 64}"
  [transformer]="{arch: transformer, layers: 4, dim: 96}"
)
TAG=$(date +%Y%m%d-%H%M)
pids=()
for arch in "${!MODELS[@]}"; do
  rlg train configs/chess_az.yaml "model=${MODELS[$arch]}" "$@" --name "chess_az_${arch}_${TAG}" > "runs/chess_az_${arch}_${TAG}.log" 2>&1 &
  pids+=($!)
  echo "started ${arch} (pid $!) -> runs/chess_az_${arch}_${TAG}.log"
done
for pid in "${pids[@]}"; do wait "$pid"; done

echo "Training finished; running the tournament..."
agents=()
for arch in "${!MODELS[@]}"; do agents+=(runs/chess_az_${arch}_${TAG}-*/best.pt); done
rlg chess-tournament "${agents[@]}" random greedy minimax2 --games 10 --sims 64 \
  --out "benchmarks/chess_arch_comparison.md"
