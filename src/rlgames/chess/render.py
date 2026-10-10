"""Turn chess games into things you can watch: a self-contained HTML replay viewer
(works in any browser, no dependencies) and an animated GIF."""

from __future__ import annotations

import html
import json
from pathlib import Path

import chess
import chess.svg
from PIL import Image, ImageDraw, ImageFont

from rlgames.chess.arena import GameRecord

_GLYPHS = {"K": "♔", "Q": "♕", "R": "♖", "B": "♗", "N": "♘", "P": "♙",
           "k": "♚", "q": "♛", "r": "♜", "b": "♝", "n": "♞", "p": "♟"}
_FONT_CANDIDATES = [
    "DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Apple Symbols.ttf",
    "C:/Windows/Fonts/seguisym.ttf",
]


def _positions(record: GameRecord) -> tuple[list[chess.Board], list[str]]:
    board = chess.Board()
    boards, sans = [board.copy()], [""]
    for move in record.board.move_stack:
        sans.append(board.san(move))
        board.push(move)
        boards.append(board.copy())
    return boards, sans


def write_html(records: list[GameRecord], out_path: str | Path, title: str = "Chess games") -> Path:
    """A page with every game: step through moves with buttons / arrow keys, or autoplay."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    games = []
    for rec in records:
        boards, sans = _positions(rec)
        frames = []
        for i, b in enumerate(boards):
            lastmove = b.peek() if i > 0 else None
            check = b.king(b.turn) if b.is_check() else None
            frames.append(chess.svg.board(b, size=420, lastmove=lastmove, check=check))
        games.append({
            "title": f"{rec.white} (white) vs {rec.black} (black): {rec.result} - {rec.reason}",
            "frames": frames, "sans": sans, "pgn": rec.pgn(),
        })
    page = _HTML_TEMPLATE.replace("__TITLE__", html.escape(title)).replace("__GAMES__", json.dumps(games))
    out_path.write_text(page)
    return out_path


def _font(size: int):
    for cand in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(cand, size)
        except OSError:
            continue
    return None


def board_image(board: chess.Board, square: int = 48, caption: str = "", subcaption: str = "") -> Image.Image:
    font = _font(int(square * 0.8))
    small = _font(13) or ImageFont.load_default()
    pad = 40
    img = Image.new("RGB", (8 * square, 8 * square + pad), (30, 30, 30))
    draw = ImageDraw.Draw(img)
    last = board.peek() if board.move_stack else None
    for sq in chess.SQUARES:
        f, r = chess.square_file(sq), chess.square_rank(sq)
        x, y = f * square, (7 - r) * square
        light = (f + r) % 2 == 1
        color = (240, 217, 181) if light else (181, 136, 99)
        if last and sq in (last.from_square, last.to_square):
            color = (205, 210, 106) if light else (170, 162, 58)
        draw.rectangle([x, y, x + square, y + square], fill=color)
        piece = board.piece_at(sq)
        if piece:
            sym = piece.symbol()
            if font is not None:
                centre = (x + square / 2, y + square / 2)
                if piece.color == chess.WHITE:  # white body under a black outline
                    draw.text(centre, _GLYPHS[sym.lower()], font=font, fill=(255, 255, 255), anchor="mm")
                draw.text(centre, _GLYPHS[sym], font=font, fill=(0, 0, 0), anchor="mm")
            else:  # no glyph font available: draw letters
                draw.text((x + square / 2 - 4, y + square / 2 - 6), sym, fill=(0, 0, 0) if sym.islower() else (255, 255, 255))
    draw.text((6, 8 * square + 3), caption, fill=(230, 230, 230), font=small)
    draw.text((6, 8 * square + 21), subcaption, fill=(230, 230, 230), font=small)
    return img


def write_gif(record: GameRecord, out_path: str | Path, ms_per_move: int = 500) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    boards, sans = _positions(record)
    frames = []
    for i, (b, san) in enumerate(zip(boards, sans)):
        move_no = f"{(i + 1) // 2}{'.' if i % 2 else '...'} {san}" if i else "start"
        frames.append(board_image(b, caption=f"{record.white} (W) vs {record.black} (B)", subcaption=move_no))
    frames.append(board_image(boards[-1], caption=f"{record.white} (W) vs {record.black} (B)",
                              subcaption=f"{record.result}  ({record.reason})"))
    durations = [ms_per_move] * (len(frames) - 1) + [3000]
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=durations, loop=0)
    return out_path


_HTML_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>__TITLE__</title>
<style>
 body { font-family: system-ui, sans-serif; background: #f4f4f4; margin: 16px; }
 .game { background: white; padding: 12px; margin-bottom: 16px; border-radius: 8px; display: inline-block; }
 .controls button { margin-right: 4px; }
 .moves { max-width: 420px; font-family: monospace; font-size: 13px; line-height: 1.6; }
 .moves span { cursor: pointer; padding: 0 2px; }
 .moves span.cur { background: #ffd54f; }
 select { margin-bottom: 12px; }
</style></head><body>
<h2>__TITLE__</h2>
<select id="pick"></select>
<div class="game">
 <div id="caption"></div>
 <div id="board"></div>
 <div class="controls">
  <button onclick="go(0)">|&lt;</button><button onclick="go(cur-1)">&lt;</button>
  <button onclick="toggle()" id="play">play</button>
  <button onclick="go(cur+1)">&gt;</button><button onclick="go(1e9)">&gt;|</button>
  <span id="ply"></span>
 </div>
 <div class="moves" id="moves"></div>
 <details><summary>PGN</summary><pre id="pgn"></pre></details>
</div>
<script>
const games = __GAMES__;
let g = 0, cur = 0, timer = null;
const pick = document.getElementById('pick');
games.forEach((x, i) => { const o = document.createElement('option'); o.value = i; o.textContent = (i + 1) + ': ' + x.title; pick.appendChild(o); });
pick.onchange = () => load(+pick.value);
function load(i) {
  g = i; stop();
  document.getElementById('caption').textContent = games[g].title;
  document.getElementById('pgn').textContent = games[g].pgn;
  const mv = document.getElementById('moves'); mv.innerHTML = '';
  games[g].sans.forEach((s, k) => { if (!k) return;
    const sp = document.createElement('span');
    sp.textContent = (k % 2 ? ((k + 1) / 2) + '. ' : '') + s; sp.onclick = () => go(k); sp.id = 'm' + k; mv.appendChild(sp); mv.append(' '); });
  go(0);
}
function go(k) {
  const n = games[g].frames.length; cur = Math.max(0, Math.min(n - 1, k));
  document.getElementById('board').innerHTML = games[g].frames[cur];
  document.getElementById('ply').textContent = 'ply ' + cur + ' / ' + (n - 1);
  document.querySelectorAll('.moves span').forEach(e => e.classList.toggle('cur', e.id === 'm' + cur));
  if (cur === n - 1) stop();
}
function stop() { clearInterval(timer); timer = null; document.getElementById('play').textContent = 'play'; }
function toggle() { if (timer) return stop(); document.getElementById('play').textContent = 'pause'; timer = setInterval(() => go(cur + 1), 600); }
document.onkeydown = e => { if (e.key === 'ArrowLeft') go(cur - 1); if (e.key === 'ArrowRight') go(cur + 1); if (e.key === ' ') { e.preventDefault(); toggle(); } };
load(0);
</script></body></html>
"""
