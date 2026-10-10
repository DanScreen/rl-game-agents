import chess
import numpy as np
import pytest
import torch

from rlgames.chess.agents import GreedyMaterialAgent, MinimaxAgent, RandomAgent, ValueNetAgent, make_baseline
from rlgames.chess.arena import elo_diff, play_game, play_match
from rlgames.chess.encoding import NUM_FEATURES, encode_board, material_balance, terminal_value
from rlgames.chess.model import ValueNet
from rlgames.chess.td import lambda_returns

# White to move, Qxf7# is mate in one (scholar's mate pattern).
MATE_IN_ONE = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4"


def test_encoding_is_side_to_move_relative():
    start = chess.Board()
    enc = encode_board(start)
    assert enc.shape == (NUM_FEATURES,)
    assert enc[: 6 * 64].sum() == 16 and enc[6 * 64: 12 * 64].sum() == 16
    # the starting position seen by black (after a null move) looks identical to white's view
    black_view = start.copy()
    black_view.push(chess.Move.null())
    np.testing.assert_array_equal(encode_board(black_view), enc)
    # our pawns are always on rank 2 from the mover's perspective
    pawns = enc[:64].reshape(8, 8)
    assert pawns[1].sum() == 8


def test_material_and_terminal_values():
    board = chess.Board(MATE_IN_ONE)
    assert material_balance(board) == 0
    board.push_san("Qxf7#")
    assert terminal_value(board) == -1.0  # side to move (black) is mated
    assert terminal_value(chess.Board()) is None


@pytest.mark.parametrize("agent", [GreedyMaterialAgent(0), MinimaxAgent(2, 0), ValueNetAgent(ValueNet(), depth=1, seed=0),
                                   ValueNetAgent(ValueNet(), depth=2, seed=0)])
def test_agents_find_mate_in_one(agent):
    board = chess.Board(MATE_IN_ONE)
    assert board.san(agent.select_move(board)) == "Qxf7#"


def test_minimax_avoids_hanging_queen():
    # Black queen on d5 is attacked by the e4 pawn; a 2-ply search must move it to safety
    board = chess.Board("rnb1kbnr/ppp1pppp/8/3q4/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 3")
    board.push(MinimaxAgent(2, 0).select_move(board))
    queen_capturable = any(
        board.is_capture(m) and board.piece_type_at(m.to_square) == chess.QUEEN for m in board.legal_moves
    )
    assert not queen_capturable


def test_lambda_returns_extremes():
    values = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    # lambda = 0: one-step negamax TD target -V(s_{t+1})
    np.testing.assert_allclose(lambda_returns(values, 1.0, 0.0), [-0.2, -0.3, -1.0], rtol=1e-6)
    # lambda = 1: Monte-Carlo, final result with alternating sign
    np.testing.assert_allclose(lambda_returns(values, 1.0, 1.0), [-1.0, 1.0, -1.0], rtol=1e-6)


def test_play_game_and_match_bookkeeping():
    rec = play_game(RandomAgent(0), RandomAgent(1), max_plies=40)
    assert rec.result in ("1-0", "0-1", "1/2-1/2")
    assert "[White \"random\"]" in rec.pgn()
    m = play_match(GreedyMaterialAgent(0), RandomAgent(1), games=4, max_plies=60)
    assert m.n == 4 and 0.0 <= m.score <= 1.0


def test_elo_diff_is_symmetric_and_finite():
    assert elo_diff(0.5, 10) == pytest.approx(0.0)
    assert elo_diff(0.75, 10) == pytest.approx(-elo_diff(0.25, 10))
    assert np.isfinite(elo_diff(1.0, 10)) and np.isfinite(elo_diff(0.0, 10))


def test_make_baseline_names():
    assert make_baseline("random").name == "random"
    assert make_baseline("greedy").name == "greedy-material"
    assert make_baseline("minimax3").name == "minimax-d3"


def test_value_net_output_range():
    net = ValueNet()
    out = net(torch.randn(5, NUM_FEATURES))
    assert out.shape == (5,) and (out.abs() <= 1).all()
