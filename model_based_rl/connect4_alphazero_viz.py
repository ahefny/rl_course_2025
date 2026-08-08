"""Gradio visualizer for a trained AlphaZero Connect-4 network.

You play both sides; after every position the UI shows the network's policy
over columns and the value of each resulting successor state.

Run:
    python model_based_rl/connect4_alphazero_viz.py path/to/checkpoint.pt

Optional:
    python model_based_rl/connect4_alphazero_viz.py path/to/checkpoint.pt --device cpu
"""

from __future__ import annotations

import argparse
import os
import sys
from functools import partial

import gradio as gr
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from connect4_alphazero import (
    AlphaZeroNet,
    net_predict,
    outcome_for,
)
from connect4_mcts import EMPTY, P1, P2, Connect4State

_DISC_COLORS = {EMPTY: "#ffffff", P1: "#e74c3c", P2: "#f1c40f"}
MAX_COLS = 12  # pre-created Gradio columns; extras stay hidden


# ===========================================================================
# Model loading
# ===========================================================================


def load_model(path: str, device: torch.device) -> tuple[AlphaZeroNet, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    args = ckpt["args"]
    rows = int(args.get("rows", 6))
    cols = int(args.get("cols", 7))
    connect = int(args.get("connect", 4))
    hidden_dim = int(args["hidden_dim"])
    blocks = int(args["blocks"])
    net = AlphaZeroNet(
        rows=rows,
        cols=cols,
        hidden_dim=hidden_dim,
        blocks=blocks,
    ).to(device)
    net.load_state_dict(ckpt["model"])
    net.eval()
    meta = {
        "rows": rows,
        "cols": cols,
        "connect": connect,
        "hidden_dim": hidden_dim,
        "blocks": blocks,
        "iteration": ckpt.get("iteration"),
        "path": path,
    }
    return net, meta


# ===========================================================================
# Network queries for the UI
# ===========================================================================


@torch.no_grad()
def evaluate_actions(
    net: AlphaZeroNet, state: Connect4State, device: torch.device,
) -> tuple[np.ndarray, list[float | None], float]:
    """Return (policy, successor_values, root_value).

    `successor_values[c]` is the network value of the state after playing
    column `c`, from the perspective of the player to move there. Illegal
    columns are None. Terminal successors use the true outcome for that
    player-to-move (+1 if they already won — which never happens — so we use
    outcome for the player who just moved, negated to match to-play view:
    if the mover won, to-play's outcome is -1).
    """
    if state.is_terminal():
        return (
            np.zeros(state.cols, dtype=np.float32),
            [None] * state.cols,
            outcome_for(state, state.to_play),  # unused; game over
        )

    policy, root_value = net_predict(net, state, device)
    values: list[float | None] = [None] * state.cols
    for col in state.get_moves():
        nxt = state.clone()
        nxt.do_move(col)
        if nxt.is_terminal():
            # Value of the resulting state from its to-play player's view.
            values[col] = outcome_for(nxt, nxt.to_play)
        else:
            _, v = net_predict(net, nxt, device)
            values[col] = v
    return policy, values, root_value


# ===========================================================================
# Rendering
# ===========================================================================


def render_board_html(state: Connect4State) -> str:
    cell, pad = 46, 6
    parts = [
        f'<div style="display:inline-block;background:#1f4e9c;padding:{pad}px;'
        'border-radius:10px;">'
    ]
    for r in range(state.rows):
        parts.append('<div style="display:flex;">')
        for c in range(state.cols):
            v = int(state.board[r, c])
            disc = "#0c2f66" if v == EMPTY else _DISC_COLORS[v]
            highlight = ""
            if state.last_move == (r, c):
                highlight = "box-shadow:0 0 0 3px #ffffff;"
            parts.append(
                f'<div style="width:{cell}px;height:{cell}px;display:flex;'
                'align-items:center;justify-content:center;">'
                f'<div style="width:{cell-12}px;height:{cell-12}px;border-radius:50%;'
                f'background:{disc};{highlight}"></div></div>'
            )
        parts.append("</div>")
    parts.append("</div>")
    return "".join(parts)


def _fmt_value(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v:+.2f}"


def render_col_stats_html(policy: np.ndarray, values: list[float | None],
                          cols: int, legal: set) -> list[str]:
    """One HTML snippet per Gradio column slot (including hidden extras).

    Policy-best (max π) is highlighted in blue. If the value-best action
    differs — lowest successor v, since v is from the opponent's view — it is
    highlighted in light gray.
    """
    best_pi = max(legal, key=lambda i: float(policy[i])) if legal else None
    # Preferred for the current player = worst for the opponent = min v(s').
    best_v = (
        min(legal, key=lambda i: float(values[i]))  # type: ignore[arg-type]
        if legal else None
    )

    out = []
    for c in range(MAX_COLS):
        if c >= cols:
            out.append("")
            continue
        if c not in legal:
            out.append(
                '<div style="text-align:center;color:#aaa;font-size:12px;'
                'font-family:monospace;line-height:1.4;">'
                'π —<br>v —</div>'
            )
            continue
        pi = float(policy[c]) * 100.0
        if c == best_pi:
            style = "font-weight:700;color:#1a5276;"  # policy-best
        elif c == best_v and best_v != best_pi:
            style = "font-weight:700;color:#a0a0a0;"  # value-best (disagree)
        else:
            style = "color:#333;"
        out.append(
            f'<div style="text-align:center;font-size:12px;font-family:monospace;'
            f'line-height:1.45;{style}">'
            f'π {pi:5.1f}%<br>v {_fmt_value(values[c])}</div>'
        )
    return out


def status_text(state: Connect4State, root_value: float) -> str:
    if state.winner == P1:
        return "### 🔴 wins"
    if state.winner == P2:
        return "### 🟡 wins"
    if state.is_terminal():
        return "### Draw"
    disc = "🔴" if state.to_play == P1 else "🟡"
    return (
        f"### {disc} to play &nbsp;·&nbsp; "
        f"V(s) = **{root_value:+.3f}** "
        f"<span style='color:#888;font-size:0.85em;'>"
        f"(value for the player to move)</span>"
    )


def button_updates(state: Connect4State):
    legal = set(state.get_moves())
    ups = []
    for c in range(MAX_COLS):
        if c < state.cols:
            ups.append(gr.update(visible=True, interactive=(c in legal)))
        else:
            ups.append(gr.update(visible=False))
    return ups


def stats_updates(policy, values, state: Connect4State):
    legal = set(state.get_moves()) if not state.is_terminal() else set()
    htmls = render_col_stats_html(policy, values, state.cols, legal)
    ups = []
    for c in range(MAX_COLS):
        if c < state.cols:
            ups.append(gr.update(value=htmls[c], visible=True))
        else:
            ups.append(gr.update(value="", visible=False))
    return ups


# ===========================================================================
# Gradio app
# ===========================================================================


def build_ui(net: AlphaZeroNet, meta: dict, device: torch.device) -> gr.Blocks:
    rows, cols, connect = meta["rows"], meta["cols"], meta["connect"]

    def fresh_sess() -> dict:
        return {"state": Connect4State(rows, cols, connect)}

    def render_all(sess: dict):
        state: Connect4State = sess["state"]
        if state.is_terminal():
            policy = np.zeros(cols, dtype=np.float32)
            values: list[float | None] = [None] * cols
            root_value = 0.0
        else:
            policy, values, root_value = evaluate_actions(net, state, device)
        return (
            sess,
            render_board_html(state),
            status_text(state, root_value),
            *button_updates(state),
            *stats_updates(policy, values, state),
        )

    def on_new_game():
        return render_all(fresh_sess())

    def on_drop(col: int, sess: dict):
        if not sess:
            sess = fresh_sess()
        state: Connect4State = sess["state"]
        if state.is_terminal() or col not in state.get_moves():
            return render_all(sess)
        state.do_move(col)
        return render_all(sess)

    iter_note = (
        f" (training iter {meta['iteration']})" if meta.get("iteration") is not None
        else ""
    )

    with gr.Blocks(title="Connect-4 AlphaZero") as demo:
        gr.Markdown(
            f"# Connect-4 · AlphaZero inspector\n"
            f"Model: `{meta['path']}`{iter_note}  \n"
            f"Board **{rows}×{cols}**, connect-{connect}, "
            f"net hidden_dim={meta['hidden_dim']} blocks={meta['blocks']}.  \n"
            "Play **both** sides. Under each column: **π** = policy probability "
            "for that action from the current state; **v** = learned value of "
            "the resulting state (from the *then*-to-play player's view)."
        )
        sess_state = gr.State()

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown(
                    "### Legend\n"
                    "- 🔴 = Player 1 &nbsp; 🟡 = Player 2\n"
                    "- **V(s)** above the board is the value for whoever is to move\n"
                    "- **v** under a column is V(s′) after dropping there\n"
                    "- <span style='color:#1a5276;font-weight:700'>Blue</span> "
                    "= best action by **π**\n"
                    "- <span style='color:#a0a0a0;font-weight:700'>Gray</span> "
                    "= best action by **v** alone (lowest successor v), shown only "
                    "when it differs from π"
                )
                new_game_btn = gr.Button("New game", variant="primary")

            with gr.Column(scale=2):
                status = gr.Markdown("### Loading…")
                # Column buttons with π / v stats directly underneath each.
                col_buttons = []
                col_stats = []
                with gr.Row():
                    for c in range(MAX_COLS):
                        with gr.Column(min_width=52, scale=0):
                            col_buttons.append(
                                gr.Button(f"▼ {c}", visible=False, min_width=48)
                            )
                            col_stats.append(
                                gr.HTML(visible=False)
                            )
                board = gr.HTML()

        outputs = [sess_state, board, status, *col_buttons, *col_stats]

        new_game_btn.click(on_new_game, inputs=[], outputs=outputs)
        for c, btn in enumerate(col_buttons):
            btn.click(partial(on_drop, c), inputs=[sess_state], outputs=outputs)
        demo.load(on_new_game, inputs=[], outputs=outputs)

    return demo


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize a trained AlphaZero Connect-4 model",
    )
    p.add_argument("model", type=str, help="Path to checkpoint .pt from training")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--share", action="store_true",
                   help="Create a public Gradio share link")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not os.path.isfile(args.model):
        raise SystemExit(f"Model not found: {args.model}")
    device = torch.device(args.device)
    net, meta = load_model(args.model, device)
    print(f"loaded {args.model} on {device}: "
          f"{meta['rows']}x{meta['cols']} connect-{meta['connect']}")
    build_ui(net, meta, device).launch(share=args.share)


if __name__ == "__main__":
    main()
