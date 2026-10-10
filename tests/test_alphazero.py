import chess
import numpy as np
import pytest
import torch

from rlgames.chess.alphazero import SelfPlayBuffer
from rlgames.chess.az_encoding import NUM_ACTIONS, NUM_PLANES, encode_planes, legal_actions, move_to_action, quick_terminal
from rlgames.chess.mcts import AlphaZeroAgent, BatchedMCTS, Node, net_evaluator
from rlgames.chess.nets import ARCHITECTURES, build_net
from rlgames.chess.tournament import fit_elo

MATE_IN_ONE = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
SMALL = {
    "mlp": {"arch": "mlp", "hidden": [32]},
    "cnn": {"arch": "cnn", "layers": 2, "channels": 8},
    "resnet": {"arch": "resnet", "blocks": 1, "channels": 8},
    "transformer": {"arch": "transformer", "layers": 1, "dim": 16, "heads": 2},
}


def test_moves_are_encoded_from_the_movers_view():
    white = chess.Board()
    e4 = move_to_action(chess.Move.from_uci("e2e4"), chess.WHITE)
    assert e4 == chess.E2 * 64 + chess.E4
    white.push_uci("e2e4")
    assert move_to_action(chess.Move.from_uci("e7e5"), chess.BLACK) == e4  # mirrored


def test_underpromotions_get_their_own_actions():
    board = chess.Board("8/1P6/8/8/8/8/8/k6K w - - 0 1")
    moves, actions = legal_actions(board)
    promo = {m.promotion: a for m, a in zip(moves, actions) if m.from_square == chess.B7}
    assert promo[chess.QUEEN] == chess.B7 * 64 + chess.B8
    assert all(4096 <= promo[p] < NUM_ACTIONS for p in (chess.KNIGHT, chess.BISHOP, chess.ROOK))
    assert len(set(actions.tolist())) == len(moves)


def test_planes_are_side_to_move_relative():
    board = chess.Board()
    p = encode_planes(board)
    assert p.shape == (NUM_PLANES, 8, 8)
    assert p[0, 1].sum() == 8 and p[6, 6].sum() == 8  # our pawns on rank 2, theirs on rank 7
    black = board.copy()
    black.push(chess.Move.null())
    np.testing.assert_array_equal(encode_planes(black)[:12], p[:12])


def test_quick_terminal():
    board = chess.Board(MATE_IN_ONE)
    board.push_san("Qxf7#")
    assert quick_terminal(board, list(board.legal_moves)) == -1.0
    assert quick_terminal(chess.Board(), list(chess.Board().legal_moves)) is None


@pytest.mark.parametrize("arch", sorted(ARCHITECTURES))
def test_every_architecture_has_the_same_interface(arch):
    net = build_net(SMALL[arch]).eval()
    logits, value = net(torch.randn(3, NUM_PLANES, 8, 8))
    assert logits.shape == (3, NUM_ACTIONS) and value.shape == (3,)
    assert (value.abs() <= 1).all()


def test_conv_policy_head_maps_squares_to_the_right_actions():
    """The a1->h8 logit is computed at the from-square (a1): it must depend on inputs near a1
    and not on inputs near h8, which lie outside the small net's receptive field from a1."""
    torch.manual_seed(0)
    net = build_net(SMALL["cnn"]).eval()  # 2 conv layers + 3x3 head conv: receptive radius 3
    x = torch.randn(1, NUM_PLANES, 8, 8, requires_grad=True)
    logits, _ = net(x)
    logits[0, chess.A1 * 64 + chess.H8].backward()
    grad = x.grad.abs().sum(dim=(0, 1))  # (8, 8) indexed [rank, file]
    assert grad[0, 0] > 0 and grad[7, 7] == 0


def test_unknown_architecture():
    with pytest.raises(ValueError):
        build_net({"arch": "nope"})


@pytest.mark.parametrize("arch", ["resnet", "transformer"])
def test_mcts_finds_mate_in_one_with_an_untrained_net(arch):
    torch.manual_seed(0)
    agent = AlphaZeroAgent(build_net(SMALL[arch]), simulations=150)
    board = chess.Board(MATE_IN_ONE)
    assert board.san(agent.select_move(board)) == "Qxf7#"


def test_batched_mcts_visits_and_restores_boards():
    mcts = BatchedMCTS(net_evaluator(build_net(SMALL["mlp"]).eval()))
    boards = [chess.Board(), chess.Board(MATE_IN_ONE)]
    fens = [b.fen() for b in boards]
    roots = [Node(), Node()]
    mcts.run(boards, roots, simulations=20)
    assert [b.fen() for b in boards] == fens
    assert all(r.N.sum() == 20 for r in roots)


def test_selfplay_buffer_round_trip():
    buf = SelfPlayBuffer(4)
    board = chess.Board()
    for _ in range(30):  # raise the fifty-move counter so plane 18 is non-trivial
        board.push(next(iter(m for m in board.legal_moves if not board.is_capture(m) and board.piece_type_at(m.from_square) == chess.KNIGHT)))
    planes = encode_planes(board)
    moves, actions = legal_actions(board)
    probs = np.full(len(moves), 1 / len(moves))
    buf.add(planes, actions, probs, 0.5)
    x, pi, z = buf.sample(2, np.random.default_rng(0))
    np.testing.assert_allclose(x[0].numpy(), planes)
    assert pi[0].sum() == pytest.approx(1.0) and z[0] == pytest.approx(0.5)


def test_fit_elo_orders_players():
    results = {(0, 1): (15.0, 20), (1, 2): (15.0, 20), (0, 2): (18.0, 20)}

    def score(i, j):
        if (i, j) in results:
            return results[(i, j)]
        s, n = results[(j, i)]
        return n - s, n

    r = fit_elo(["a", "b", "random"], score)
    assert r[2] == 0 and r[0] > r[1] > r[2]


def test_curriculum_start_positions_are_legal_and_reduced():
    from rlgames.chess.alphazero import DEFAULTS, start_position

    rng = np.random.default_rng(0)
    hp = {**DEFAULTS, "curriculum_frac": 1.0, "curriculum_keep": [0.2, 0.5], "opening_random_plies": [0, 4]}
    for _ in range(50):
        board = start_position(rng, hp)
        assert board.is_valid() and not board.is_game_over(claim_draw=True)
        assert len(board.piece_map()) < 32 and board.king(chess.WHITE) is not None and board.king(chess.BLACK) is not None
    plain = start_position(rng, DEFAULTS)
    assert plain.fen() == chess.STARTING_FEN
