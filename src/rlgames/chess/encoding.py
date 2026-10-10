"""Board -> feature vector, always from the perspective of the side to move.

Black-to-move positions are mirrored (ranks flipped, colours swapped) so the network
only ever sees "we are white, it is our move". That halves what it has to learn and
makes ``V(position)`` a negamax value: +1 = side to move wins, -1 = it loses.

Features (772): 12 x 64 piece-occupancy planes (our P N B R Q K, then theirs) +
4 castling-right flags (our K/Q side, their K/Q side).
"""

from __future__ import annotations

import chess
import numpy as np

NUM_FEATURES = 12 * 64 + 4

_PIECE_TYPES = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)
PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


def _bitboards(board: chess.Board) -> tuple[np.ndarray, tuple[bool, bool, bool, bool]]:
    """12 piece bitboards (ours then theirs) and 4 castling flags, side to move = white."""
    b = board if board.turn == chess.WHITE else board.mirror()
    words = [b.pieces_mask(pt, chess.WHITE) for pt in _PIECE_TYPES]
    words += [b.pieces_mask(pt, chess.BLACK) for pt in _PIECE_TYPES]
    return np.array(words, dtype=np.uint64), (
        b.has_kingside_castling_rights(chess.WHITE),
        b.has_queenside_castling_rights(chess.WHITE),
        b.has_kingside_castling_rights(chess.BLACK),
        b.has_queenside_castling_rights(chess.BLACK),
    )


def encode_boards(boards: list[chess.Board]) -> np.ndarray:
    out = np.zeros((len(boards), NUM_FEATURES), dtype=np.float32)
    if not boards:
        return out
    words = np.empty((len(boards), 12), dtype=np.uint64)
    castling = np.empty((len(boards), 4), dtype=np.float32)
    for i, board in enumerate(boards):
        words[i], castling[i] = _bitboards(board)
    # unpack each 64-bit board into 64 bits, square 0 (a1) first
    bits = np.unpackbits(words.astype("<u8").view(np.uint8).reshape(len(boards), 12, 8), axis=2, bitorder="little")
    out[:, : 12 * 64] = bits.reshape(len(boards), 12 * 64)
    out[:, 12 * 64:] = castling
    return out


def encode_board(board: chess.Board) -> np.ndarray:
    return encode_boards([board])[0]


def material_balance(board: chess.Board) -> int:
    """Material (pawn units) of the side to move minus the opponent's."""
    total = 0
    for pt, v in PIECE_VALUES.items():
        total += v * (len(board.pieces(pt, board.turn)) - len(board.pieces(pt, not board.turn)))
    return total


def terminal_value(board: chess.Board, claim_draw: bool = False) -> float | None:
    """Exact value for the side to move if the game is over, else None."""
    outcome = board.outcome(claim_draw=claim_draw)
    if outcome is None:
        return None
    if outcome.winner is None:
        return 0.0
    return 1.0 if outcome.winner == board.turn else -1.0
