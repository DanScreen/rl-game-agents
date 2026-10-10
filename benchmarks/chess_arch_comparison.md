# Chess tournament

10 games per pair, colours alternating, 4 random opening plies. Elo is anchored at random = 0.

| # | Agent | Elo | Score | #1 | #2 | #3 | #4 | #5 | #6 | #7 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | minimax-d2 | +353 | 87% | - | 10/10 | 6/10 | 8/10 | 9/10 | 9/10 | 10/10 |
| 2 | greedy-material | +258 | 75% | 0/10 | - | 7/10 | 10/10 | 9/10 | 9.5/10 | 9.5/10 |
| 3 | chess_az_transformer_20261010-0712-20261010-071253/best-s64 | +57 | 44% | 4/10 | 3/10 | - | 5/10 | 5/10 | 5/10 | 4.5/10 |
| 4 | chess_az_mlp_20261010-0712-20261010-071254/best-s64 | +11 | 37% | 2/10 | 0/10 | 5/10 | - | 5/10 | 5/10 | 5/10 |
| 5 | chess_az_cnn_20261010-0712-20261010-071253/best-s64 | +11 | 37% | 1/10 | 1/10 | 5/10 | 5/10 | - | 5/10 | 5/10 |
| 6 | chess_az_resnet_20261010-0712-20261010-071253/best-s64 | +5 | 36% | 1/10 | 0.5/10 | 5/10 | 5/10 | 5/10 | - | 5/10 |
| 7 | random | +0 | 35% | 0/10 | 0.5/10 | 5.5/10 | 5/10 | 5/10 | 5/10 | - |

## Findings (2026-10-10)

All four architectures were trained with `configs/chess_az.yaml` (2,000 pure self-play games,
32 simulations per move, one CPU core each). None got off the ground:

* Early self-play is near-random and decisive (~60% of the first 100 games end in checkmate).
  Within ~400 games both sides learn to avoid losing faster than they learn to win: checkmates
  drop to 1-8% and games end in threefold repetition after ~50 plies.
* The value head ends up predicting ~0 everywhere (it rates a queen up, a queen down and
  K+Q vs K all at about 0), so the search cannot steer towards wins and the loop stalls.
* With the networks this undertrained, the architecture ranking above is noise.

Follow-up pilots (ResNet, 600 games each):

| Pilot | Change | vs random | vs greedy |
|---|---|---|---|
| curriculum | 50% of games start with random pieces removed, 60 plies of move sampling | 42.5% | 17.5% |
| A (pure) | 80% lopsided endgame starts, 64 simulations | 55% | 10% |
| B (hint) | as curriculum + unfinished games scored by material | 47.5% | 10% |

Sanity check: the same network, encoding, buffer and loss trained *supervised* on material
labels learns them quickly (value error 0.49 -> 0.02; +0.56 a queen up, -0.74 a queen down,
symmetric for either side to move). The pipeline works; the self-play signal at this scale
does not carry enough information. AlphaZero escaped this regime with tens of millions of
games; this is the classic cold-start problem of learning from zero with little compute.
