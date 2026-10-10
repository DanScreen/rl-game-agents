"""Board and move encoding for policy + value networks (AlphaZero-style).

Everything is from the side to move's perspective: black-to-move positions are
mirrored (ranks flipped, colours swapped), so the network always plays "white".

Input: 20 planes of 8x8 (row = rank, column = file, after mirroring)
  0-5    our pawns, knights, bishops, rooks, queens, king
  6-11   their pieces, same order
  12     position has occurred before (repetition)
  13-16  castling rights: our king/queen side, their king/queen side
  17     en-passant target square
  18     fifty-move counter / 100
  19     constant ones (lets convolutions see the board edge)

Policy: 4168 logits
  0..4095     from_square * 64 + to_square   (includes queen promotions)
  4096..4167  under-promotions: (from_file * 3 + direction) * 3 + piece
              direction = to_file - from_file + 1 (0, 1, 2), piece = knight/bishop/rook
"""

from __future__ import annotations

import chess
import numpy as np

NUM_PLANES = 20
NUM_ACTIONS = 64 * 64 + 8 * 3 * 3  # 4168

_PIECE_TYPES = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)
_UNDERPROMO = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2}


def move_to_action(move: chess.Move, turn: chess.Color) -> int:
    f, t = move.from_square, move.to_square
    if turn == chess.BLACK:
        f, t = chess.square_mirror(f), chess.square_mirror(t)
    if move.promotion is not None and move.promotion != chess.QUEEN:
        direction = chess.square_file(t) - chess.square_file(f) + 1
        return 4096 + (chess.square_file(f) * 3 + direction) * 3 + _UNDERPROMO[move.promotion]
    return f * 64 + t


def legal_actions(board: chess.Board, moves: list[chess.Move] | None = None) -> tuple[list[chess.Move], np.ndarray]:
    moves = list(board.legal_moves) if moves is None else moves
    return moves, np.fromiter((move_to_action(m, board.turn) for m in moves), dtype=np.int64, count=len(moves))


def encode_planes(board: chess.Board) -> np.ndarray:
    planes = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)
    repeated = board.is_repetition(2)  # needs the original move stack, so check before mirroring
    b = board if board.turn == chess.WHITE else board.mirror()
    words = np.array([b.pieces_mask(pt, chess.WHITE) for pt in _PIECE_TYPES]
                     + [b.pieces_mask(pt, chess.BLACK) for pt in _PIECE_TYPES], dtype="<u8")
    bits = np.unpackbits(words.view(np.uint8).reshape(12, 8), axis=1, bitorder="little")
    planes[:12] = bits.reshape(12, 8, 8)  # square = rank * 8 + file -> [rank, file]
    if repeated:
        planes[12] = 1.0
    planes[13] = float(b.has_kingside_castling_rights(chess.WHITE))
    planes[14] = float(b.has_queenside_castling_rights(chess.WHITE))
    planes[15] = float(b.has_kingside_castling_rights(chess.BLACK))
    planes[16] = float(b.has_queenside_castling_rights(chess.BLACK))
    if b.ep_square is not None:
        planes[17, chess.square_rank(b.ep_square), chess.square_file(b.ep_square)] = 1.0
    planes[18] = min(board.halfmove_clock, 100) / 100.0
    planes[19] = 1.0
    return planes


def quick_terminal(board: chess.Board, moves: list[chess.Move]) -> float | None:
    """Exact value for the side to move if the game is over, else None.

    Cheaper than ``board.outcome(claim_draw=True)``: reuses the legal move list and treats
    threefold repetition and the fifty-move rule as draws (as if claimed).
    """
    if not moves:
        return -1.0 if board.is_check() else 0.0
    if board.halfmove_clock >= 100 or board.is_insufficient_material() or board.is_repetition(3):
        return 0.0
    return None
