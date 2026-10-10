"""Command-line interface: ``rlg <command> ...``. Run ``rlg -h`` for the list."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


# ----------------------------------------------------------------------------- train
def cmd_train(args: argparse.Namespace) -> None:
    from rlgames.config import load_config
    from rlgames.logger import make_run_dir, seed_everything

    cfg = load_config(args.config, args.overrides)
    if args.name:
        cfg["name"] = args.name
    torch.set_num_threads(int(cfg.get("threads", 1)))
    seed_everything(int(cfg.get("seed", 0)))
    algo = cfg.get("algo", {}).get("name")
    run_dir = make_run_dir(cfg, args.runs_dir)
    print(f"Run directory: {run_dir}", flush=True)
    if algo == "dqn":
        from rlgames.agents.dqn import train
    elif algo == "ppo":
        from rlgames.agents.ppo import train
    elif algo == "chess_td":
        from rlgames.chess.td import train
    elif algo == "chess_az":
        from rlgames.chess.alphazero import train
    else:
        sys.exit(f"Unknown algo.name {algo!r} in {args.config} (expected dqn, ppo, chess_td or chess_az)")
    summary = train(cfg, run_dir)
    print(json.dumps({k: v for k, v in summary.items() if k != "eval_history"}, indent=2))
    print(f"Done. Checkpoints: {run_dir}/best.pt and {run_dir}/latest.pt")


# ----------------------------------------------------------------------------- atari / gym agents
def _load_agent(spec: str, env: str | None, seed: int = 0, epsilon: float | None = None):
    """``spec`` is a checkpoint / run dir, or ``random`` (needs --env)."""
    from rlgames.agents.policy import RandomPolicy, load_policy
    from rlgames.envs import make_env

    if spec == "random":
        if not env:
            sys.exit("--env is required for the random agent")
        e = make_env(env)
        n = int(e.action_space.n)
        e.close()
        return RandomPolicy(env, n, seed=seed)
    return load_policy(spec, epsilon=epsilon, seed=seed)


def cmd_watch(args: argparse.Namespace) -> None:
    from rlgames.record import play_human, record_episodes

    policy = _load_agent(args.agent, args.env, args.seed, args.epsilon)
    env = args.env or policy.env
    if args.human:
        play_human(policy, env, args.episodes, args.seed, policy.env_kwargs)
        return
    out = Path(args.out or f"videos/{env.replace('/', '_')}-{policy.name.replace('/', '_')}.mp4")
    scores = record_episodes(policy, env, out, args.episodes, args.seed, policy.env_kwargs, fps=args.fps,
                             max_steps=args.max_steps)
    print(f"Scores: {scores}")
    print(f"Saved video: {out}")


def cmd_bench(args: argparse.Namespace) -> None:
    from rlgames.benchmark import record_result
    from rlgames.evaluate import run_episodes

    agents = [_load_agent(spec, args.env, args.seed, args.epsilon) for spec in args.agents]
    if args.with_random:
        agents.append(_load_agent("random", args.env or agents[0].env, args.seed))
    print(f"{'agent':40s} {'mean':>9s} {'std':>8s} {'median':>8s} {'min':>7s} {'max':>7s}")
    for policy in agents:
        env = args.env or policy.env
        stats = run_episodes(policy, env, args.episodes, seed=args.seed + 10_000, env_kwargs=policy.env_kwargs)
        print(f"{policy.name:40s} {stats['mean_return']:9.1f} {stats['std_return']:8.1f} "
              f"{stats['median_return']:8.1f} {stats['min_return']:7.0f} {stats['max_return']:7.0f}")
        if not args.no_record:
            record_result({"game": env, "agent": policy.name, "train_steps": getattr(policy, "step", None),
                           **{k: v for k, v in stats.items() if k != "returns"}})
    if not args.no_record:
        print("Updated benchmarks/LEADERBOARD.md")


# ----------------------------------------------------------------------------- chess
def _chess_agent(spec: str, depth: int, seed: int, sims: int = 100):
    """A trained checkpoint (TD value net -> ``depth``-ply search; AlphaZero net -> MCTS with
    ``sims`` simulations) or a baseline name."""
    from rlgames.agents.policy import init_checkpoint_path
    from rlgames.chess.agents import ValueNetAgent, make_baseline

    if Path(spec).exists():
        p = Path(spec)
        label = p.name if p.is_dir() else f"{p.parent.name}/{p.stem}"
        ckpt = torch.load(init_checkpoint_path(p), map_location="cpu", weights_only=False)
        if ckpt.get("algo") == "chess_az":
            from rlgames.chess.alphazero import load_az_net
            from rlgames.chess.mcts import AlphaZeroAgent

            net, _ = load_az_net(spec)
            return AlphaZeroAgent(net, simulations=sims, name=f"{label}-s{sims}")
        from rlgames.chess.model import load_value_net

        net, _ = load_value_net(spec)
        return ValueNetAgent(net, depth=depth, seed=seed, name=f"{label}-d{depth}")
    return make_baseline(spec, seed)


def cmd_chess_bench(args: argparse.Namespace) -> None:
    from rlgames.benchmark import record_result
    from rlgames.chess.arena import play_match
    from rlgames.chess.render import write_html

    agent = _chess_agent(args.agent, args.depth, args.seed, args.sims)
    for opp_spec in args.opponents:
        opponent = _chess_agent(opp_spec, args.depth, args.seed + 1, args.sims)

        def progress(i, rec):
            print(f"  game {i + 1}/{args.games}: {rec.white} vs {rec.black} {rec.result} ({rec.reason}, {rec.board.ply()} plies)", flush=True)

        m = play_match(agent, opponent, games=args.games, max_plies=args.max_plies, opening_plies=args.opening_plies,
                       seed=args.seed, on_game=progress if args.verbose else None)
        print(m, f"[{m.seconds:.0f}s]")
        if not args.no_record:
            record_result({"game": "chess", **m.summary()})
        if args.save_games:
            out = Path(args.save_games) / f"{agent.name.replace('/', '_')}-vs-{opponent.name}.html"
            write_html(m.games, out, title=str(m))
            print(f"  games saved to {out}")
        if hasattr(opponent, "close"):
            opponent.close()


def cmd_chess_watch(args: argparse.Namespace) -> None:
    import random

    from rlgames.chess.arena import play_game
    from rlgames.chess.render import write_gif, write_html

    white = _chess_agent(args.white, args.depth, args.seed, args.sims)
    black = _chess_agent(args.black, args.depth, args.seed + 1, args.sims)
    records = []
    for g in range(args.games):
        rec = play_game(white, black, max_plies=args.max_plies, opening_plies=args.opening_plies, rng=random.Random(args.seed + g))
        print(f"game {g + 1}: {rec.white} vs {rec.black}: {rec.result} ({rec.reason}, {rec.board.ply()} plies)")
        records.append(rec)
    out = Path(args.out or f"videos/chess-{white.name.replace('/', '_')}-vs-{black.name.replace('/', '_')}.html")
    write_html(records, out, title=f"{white.name} vs {black.name}")
    print(f"Replay viewer: {out}  (open in a browser)")
    if args.gif:
        gif = write_gif(records[0], out.with_suffix(".gif"))
        print(f"GIF of game 1: {gif}")


def cmd_chess_play(args: argparse.Namespace) -> None:
    import chess

    agent = _chess_agent(args.agent, args.depth, args.seed, args.sims)
    board = chess.Board()
    human_color = chess.WHITE if args.color == "white" else chess.BLACK
    print(f"You are {args.color}. Enter moves in SAN (e4, Nf3, O-O) or UCI (e2e4). 'quit' to stop.")
    while not board.is_game_over(claim_draw=True):
        print("\n" + (str(board) if human_color == chess.WHITE else str(board.transform(chess.flip_vertical).transform(chess.flip_horizontal))))
        if board.turn == human_color:
            text = input("your move> ").strip()
            if text in ("quit", "exit"):
                return
            try:
                move = board.parse_san(text)
            except ValueError:
                try:
                    move = chess.Move.from_uci(text)
                    if move not in board.legal_moves:
                        raise ValueError
                except ValueError:
                    print("illegal move")
                    continue
        else:
            move = agent.select_move(board)
            print(f"{agent.name} plays {board.san(move)}")
        board.push(move)
    print(board)
    print(f"Game over: {board.outcome(claim_draw=True).result()}")


def cmd_chess_tournament(args: argparse.Namespace) -> None:
    from rlgames.benchmark import record_result
    from rlgames.chess.tournament import run_tournament

    agents = [_chess_agent(spec, args.depth, args.seed + i, args.sims) for i, spec in enumerate(args.agents)]
    names = [a.name for a in agents]
    if len(set(names)) != len(names):
        sys.exit(f"Agent names must be unique, got {names}")

    def progress(m):
        print(m, f"[{m.seconds:.0f}s]", flush=True)
        if not args.no_record:
            record_result({"game": "chess", **m.summary()})

    result = run_tournament(agents, games_per_pair=args.games, max_plies=args.max_plies,
                            opening_plies=args.opening_plies, seed=args.seed, on_match=progress)
    table = result.table()
    print()
    print(table)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(f"# Chess tournament\n\n{args.games} games per pair, colours alternating, "
                       f"{args.opening_plies} random opening plies. Elo is anchored at random = 0.\n\n{table}\n")
        print(f"\nSaved {out}")


# ----------------------------------------------------------------------------- plots
def cmd_plot(args: argparse.Namespace) -> None:
    from rlgames.plot import plot_runs

    runs = [Path(r) for r in args.runs]
    missing = [str(r) for r in runs if not (r / "metrics.csv").exists()]
    if missing:
        sys.exit(f"No metrics.csv in: {', '.join(missing)}")
    out = plot_runs(runs, Path(args.out), args.metric, args.title)
    print(f"Saved plot: {out}")


# ----------------------------------------------------------------------------- leaderboard
def cmd_leaderboard(args: argparse.Namespace) -> None:
    from rlgames.benchmark import LEADERBOARD, write_leaderboard

    write_leaderboard()
    print(LEADERBOARD.read_text())


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rlg", description="Train, watch and benchmark RL game agents.")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("train", help="launch a training run from a YAML config")
    s.add_argument("config")
    s.add_argument("overrides", nargs="*", help="key=value config overrides, e.g. total_steps=50000 algo.lr=3e-4")
    s.add_argument("--name", help="run name (default: config file name)")
    s.add_argument("--runs-dir", default="runs")
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("watch", help="record a video of an agent playing (or show it live with --human)")
    s.add_argument("agent", help="checkpoint file, run directory, or 'random'")
    s.add_argument("--env", help="environment (default: the one the agent was trained on)")
    s.add_argument("--episodes", type=int, default=1)
    s.add_argument("--out", help="output .mp4 or .gif (default: videos/<env>-<agent>.mp4)")
    s.add_argument("--fps", type=int)
    s.add_argument("--max-steps", type=int, default=20_000, help="cut each episode after this many steps")
    s.add_argument("--epsilon", type=float, help="DQN exploration during play (default from checkpoint)")
    s.add_argument("--human", action="store_true", help="open a live window instead of recording")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_watch)

    s = sub.add_parser("bench", help="evaluate agents over many episodes and update the leaderboard")
    s.add_argument("agents", nargs="+", help="checkpoint files / run directories / 'random'")
    s.add_argument("--env")
    s.add_argument("--episodes", type=int, default=30)
    s.add_argument("--epsilon", type=float)
    s.add_argument("--with-random", action="store_true", help="also benchmark a uniformly random agent")
    s.add_argument("--no-record", action="store_true", help="don't write to benchmarks/")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_bench)

    agent_help = ("checkpoint / run dir of a trained network (TD value net or AlphaZero net), "
                  "or a baseline: random, greedy, minimax2, minimax3, stockfish0-20")
    s = sub.add_parser("chess-bench", help="play a chess agent against opponents and update the leaderboard")
    s.add_argument("agent", help=agent_help)
    s.add_argument("--opponents", nargs="+", default=["random", "greedy", "minimax2"])
    s.add_argument("--games", type=int, default=20)
    s.add_argument("--depth", type=int, default=1, help="search depth for TD value-net agents (1, 2, 3+)")
    s.add_argument("--sims", type=int, default=100, help="MCTS simulations per move for AlphaZero agents")
    s.add_argument("--max-plies", type=int, default=300)
    s.add_argument("--opening-plies", type=int, default=4, help="random opening moves so games differ")
    s.add_argument("--save-games", metavar="DIR", help="write an HTML replay of every game to DIR")
    s.add_argument("--no-record", action="store_true")
    s.add_argument("--verbose", "-v", action="store_true")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_chess_bench)

    s = sub.add_parser("chess-watch", help="play chess games between two agents and write an HTML replay (and GIF)")
    s.add_argument("white", help=agent_help)
    s.add_argument("black", help=agent_help)
    s.add_argument("--games", type=int, default=1)
    s.add_argument("--depth", type=int, default=1)
    s.add_argument("--sims", type=int, default=100)
    s.add_argument("--max-plies", type=int, default=300)
    s.add_argument("--opening-plies", type=int, default=0)
    s.add_argument("--out", help="output .html (default: videos/chess-<white>-vs-<black>.html)")
    s.add_argument("--gif", action="store_true", help="also write an animated GIF of the first game")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_chess_watch)

    s = sub.add_parser("chess-tournament", help="round-robin between chess agents with Elo ratings")
    s.add_argument("agents", nargs="+", help=agent_help)
    s.add_argument("--games", type=int, default=10, help="games per pair")
    s.add_argument("--depth", type=int, default=1)
    s.add_argument("--sims", type=int, default=100)
    s.add_argument("--max-plies", type=int, default=300)
    s.add_argument("--opening-plies", type=int, default=4)
    s.add_argument("--out", help="write the cross-table to this markdown file")
    s.add_argument("--no-record", action="store_true")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_chess_tournament)

    s = sub.add_parser("chess-play", help="play against an agent in the terminal")
    s.add_argument("agent", help=agent_help)
    s.add_argument("--color", choices=["white", "black"], default="white")
    s.add_argument("--depth", type=int, default=2)
    s.add_argument("--sims", type=int, default=200)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_chess_play)

    s = sub.add_parser("plot", help="plot learning curves of one or more runs to a PNG")
    s.add_argument("runs", nargs="+", help="run directories")
    s.add_argument("--metric", "-m", action="append",
                   help="metric column(s) from metrics.csv, repeatable (default: eval/mean_return or eval/mean_score)")
    s.add_argument("--out", default="videos/learning_curves.png")
    s.add_argument("--title")
    s.set_defaults(func=cmd_plot)

    s = sub.add_parser("leaderboard", help="regenerate and print benchmarks/LEADERBOARD.md")
    s.set_defaults(func=cmd_leaderboard)
    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args, extra = parser.parse_known_args(argv)
    if extra:
        # allow `rlg train cfg.yaml --name x lr=1` (overrides after options)
        if args.command == "train" and all("=" in e and not e.startswith("-") for e in extra):
            args.overrides = [*args.overrides, *extra]
        else:
            parser.error(f"unrecognized arguments: {' '.join(extra)}")
    args.func(args)


if __name__ == "__main__":
    main()
