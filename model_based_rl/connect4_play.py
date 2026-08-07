"""Interactive Connect-4 played against an MCTS agent, with a Gradio UI.

The search tree built by MCTS on the computer's turn is visualized with
Graphviz; every node shows the board state it represents plus its visit
count and value estimate.

Run (classic UCT with random rollouts):
    python model_based_rl/connect4_play.py

Run (AlphaZero PUCT-MCTS with a trained checkpoint):
    python model_based_rl/connect4_play.py --checkpoint path/to/connect4_az.pt
    python model_based_rl/connect4_play.py path/to/connect4_az.pt

Open the browser to http://localhost:7860/

Requirements:
    pip install -r model_based_rl/requirements.txt     # Python packages
    sudo apt install graphviz           # the `dot` binary used to render the tree

Game rules and classic MCTS live in connect4_mcts.py; AlphaZero search lives in
connect4_alphazero.py. This file is presentation only.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from dataclasses import dataclass
from functools import partial
from typing import Any, List, Optional, Sequence, Tuple

import gradio as gr
import torch

# Allow `python model_based_rl/connect4_play.py` from the repo root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from connect4_alphazero import AZMCTS, AlphaZeroNet  # noqa: E402
from connect4_mcts import (  # noqa: E402
    EMPTY,
    P1,
    P2,
    Connect4State,
    best_move,
    child_by_move,
    mcts_search,
)


# ===========================================================================
# Play configuration (set once from the CLI before the UI launches)
# ===========================================================================


@dataclass
class PlayConfig:
    az_mode: bool = False
    net: Optional[AlphaZeroNet] = None
    device: Optional[torch.device] = None
    rows: int = 6
    cols: int = 7
    connect: int = 4
    checkpoint_path: Optional[str] = None
    iteration: Optional[int] = None


CFG = PlayConfig()


def load_az_checkpoint(path: str, device: torch.device) -> Tuple[AlphaZeroNet, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    args = ckpt["args"]
    rows = int(args.get("rows", 6))
    cols = int(args.get("cols", 7))
    connect = int(args.get("connect", 4))
    hidden_dim = int(args["hidden_dim"])
    blocks = int(args["blocks"])
    net = AlphaZeroNet(
        rows=rows, cols=cols, hidden_dim=hidden_dim, blocks=blocks,
    ).to(device)
    net.load_state_dict(ckpt["model"])
    net.eval()
    meta = {
        "rows": rows,
        "cols": cols,
        "connect": connect,
        "iteration": ckpt.get("iteration"),
        "path": path,
    }
    return net, meta


# ===========================================================================
# Tree helpers (classic Node and AlphaZero AZNode)
# ===========================================================================


def _visits(node: Any) -> int:
    return int(getattr(node, "N", getattr(node, "visits", 0)))


def _raw_q(node: Any) -> float:
    if hasattr(node, "Q"):
        return float(node.Q)
    return float(node.q)


def _state_value(node: Any, is_root: bool) -> float:
    """Estimated value of `node`'s state from the player-to-move's view.

    Classic UCT stores win-rate in [0, 1] for the player who just moved, so the
    to-play value is `1 - q`. AlphaZero stores Q in [-1, 1] for the mover into
    the node; the root's backed-up Q is already in the to-play frame.
    """
    q = _raw_q(node)
    if CFG.az_mode:
        return q if is_root else -q
    return 1.0 - q


def _child_list(node: Any) -> List[Tuple[int, Any]]:
    children = node.children
    if isinstance(children, dict):
        return list(children.items())
    return [(ch.move, ch) for ch in children]


def count_tree_nodes(node: Any) -> int:
    return 1 + sum(count_tree_nodes(ch) for _, ch in _child_list(node))


def az_child_by_move(node: Any, move: int) -> Optional[Any]:
    children = node.children
    if isinstance(children, dict):
        return children.get(move)
    return child_by_move(node, move)


# ===========================================================================
# 1. TREE VISUALIZATION (Graphviz)
# ===========================================================================
# Each node is drawn as an HTML-like table: a mini colored board on top and
# "N=.. V=.." stats below. Edges are labeled with the column played and colored
# by the mover. To stay readable, the drawing is limited by depth and, at each
# level, to the top-k most-visited children.

_DISC_COLORS = {EMPTY: "#ffffff", P1: "#e74c3c", P2: "#f1c40f"}
# Shades used to highlight the tile placed by the move leading to a node:
# a darker red and a more saturated yellow.
_LAST_DISC_COLORS = {P1: "#ff0000", P2: "#ffff00"}
_EDGE_COLORS = {P1: "#e74c3c", P2: "#d4ac0d"}


def _cell_color(state: Connect4State, r: int, c: int) -> str:
    """Disc color for (r, c); the tile just placed gets a lighter highlight shade."""
    v = int(state.board[r, c])
    if v != EMPTY and state.last_move == (r, c):
        return _LAST_DISC_COLORS[v]
    return _DISC_COLORS[v]


def _board_table_html(state: Connect4State) -> str:
    """Mini board as a Graphviz HTML-like <TABLE>."""
    rows_html = []
    for r in range(state.rows):
        cells = "".join(
            f'<TD BGCOLOR="{_cell_color(state, r, c)}" '
            f'WIDTH="11" HEIGHT="11" FIXEDSIZE="TRUE"> </TD>'
            for c in range(state.cols)
        )
        rows_html.append(f"<TR>{cells}</TR>")
    return (
        '<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" COLOR="#5d6d7e">'
        + "".join(rows_html)
        + "</TABLE>"
    )


def _node_label(node: Any, state: Connect4State, is_root: bool,
                is_best: bool) -> str:
    """Full HTML-like label (must be wrapped in <...> for Graphviz)."""
    border_color = "#2c3e50" if is_root else ("#27ae60" if is_best else "#bdc3c7")
    bg = "#eaf2ff" if is_root else ("#eafaf1" if is_best else "white")
    v = _state_value(node, is_root)
    stats = f"N={_visits(node)} &nbsp; V={v:.2f}"
    inner = (
        f'<TABLE BORDER="2" CELLBORDER="0" CELLSPACING="0" CELLPADDING="3" '
        f'COLOR="{border_color}" BGCOLOR="{bg}">'
        f"<TR><TD>{_board_table_html(state)}</TD></TR>"
        f'<TR><TD><FONT POINT-SIZE="9">{stats}</FONT></TD></TR>'
        "</TABLE>"
    )
    return f"<{inner}>"


def build_tree_graph(root: Any, root_state: Connect4State, max_depth: int,
                     top_k: int):
    """Return a graphviz.Digraph of the search tree (boards reconstructed by replay)."""
    import graphviz  # imported lazily so the module imports even without it

    dot = graphviz.Digraph("mcts")
    dot.attr(rankdir="TB", bgcolor="transparent")
    dot.attr("node", shape="plaintext", fontname="Helvetica")
    dot.attr("edge", fontname="Helvetica", fontsize="10")

    ids = itertools.count()

    def recurse(node: Any, state: Connect4State, depth: int, node_id: str,
                is_root: bool, is_best: bool) -> None:
        dot.node(node_id, label=_node_label(node, state, is_root, is_best))
        kids = _child_list(node)
        if depth >= max_depth or not kids:
            return
        shown = sorted(kids, key=lambda mv_ch: _visits(mv_ch[1]), reverse=True)[:top_k]
        best_move_ch = max(kids, key=lambda mv_ch: _visits(mv_ch[1]))[1]
        for move, ch in shown:
            child_id = str(next(ids))
            child_state = state.clone()
            child_state.do_move(move)
            recurse(ch, child_state, depth + 1, child_id, False, ch is best_move_ch)
            mover = child_state.player_just_moved
            dot.edge(
                node_id,
                child_id,
                label=f" {move}",
                color=_EDGE_COLORS.get(mover, "#888888"),
                penwidth="2.2" if ch is best_move_ch else "1.0",
            )

    recurse(root, root_state, 0, str(next(ids)), True, False)
    return dot


# ===========================================================================
# 2. GRADIO UI
# ===========================================================================
# Everything below is presentation. It reads/writes a per-session `sess` dict
# (held in gr.State) that bundles the current game state and the last MCTS root.

MAX_ROWS, MAX_COLS = 12, 12  # slider caps == number of pre-created column buttons


# -- session helpers --------------------------------------------------------
def computer_think(sess: dict, n_iter: int, C: float,
                   reuse_root: Optional[Any] = None) -> None:
    """Run MCTS for the computer, store the tree, and play the chosen move.

    Classic mode may warm-start from `reuse_root`. AlphaZero mode always builds
    a fresh PUCT tree (network priors + values, no rollouts, no Dirichlet noise).
    """
    state: Connect4State = sess["state"]
    if state.is_terminal():
        return
    root_state = state.clone()

    if CFG.az_mode:
        assert CFG.net is not None and CFG.device is not None
        az = AZMCTS(
            CFG.net, CFG.device,
            n_sims=int(n_iter),
            c_puct=float(C),
            dirichlet_eps=0.0,
        )
        policy, root = az.run(root_state, add_noise=False)
        move = int(policy.argmax())
    else:
        root = mcts_search(
            root_state, int(n_iter), float(C), root=reuse_root,
        )
        move = best_move(root)

    sess["root"] = root
    sess["root_state"] = root_state
    sess["last_computer_move"] = move
    state.do_move(move)


# -- rendering --------------------------------------------------------------
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


def status_text(sess: dict) -> str:
    state: Connect4State = sess["state"]
    human = sess["human"]
    h_disc = "🔴" if human == P1 else "🟡"
    if state.winner != 0:
        return ("### 🏆 You win!" if state.winner == human
                else "### 🤖 Computer wins.")
    if state.is_terminal():
        return "### 🤝 Draw."
    return f"### Your turn ({h_disc}) — pick a column."


def render_tree_html(sess: dict, depth: int, top_k: int) -> str:
    root = sess.get("root")
    if not root:
        return ('<div style="padding:24px;color:#888;">The search tree appears '
                "here after the computer's first move.</div>")
    reuse_note = (
        "" if CFG.az_mode else
        ' <span style="color:#999;">(exceeds the budget when tree reuse is on)</span>'
    )
    value_note = (
        "V = backed-up state value for the side to move"
        + (" ([-1, 1])" if CFG.az_mode else " ([0, 1])")
    )
    caption = (
        f'<div style="color:#555;font-size:13px;margin-bottom:6px;">'
        f"Full tree: <b>{count_tree_nodes(root)}</b> nodes, root visits "
        f"<b>N={_visits(root)}</b>{reuse_note}<br>"
        f'<span style="color:#777;">{value_note}</span>'
        f"</div>"
    )
    try:
        dot = build_tree_graph(root, sess["root_state"], int(depth), int(top_k))
        svg = dot.pipe(format="svg").decode("utf-8")
    except Exception as e:  # dot binary missing, etc.
        return caption + (
            '<div style="color:#c0392b;padding:16px;">Could not render the tree: '
            f"<code>{e}</code><br>If the Graphviz binary is missing, install it with "
            "<code>sudo apt install graphviz</code>.</div>"
        )
    return caption + (
        '<div style="overflow:auto;max-height:640px;border:1px solid #e0e0e0;'
        f'border-radius:8px;padding:8px;background:white;">{svg}</div>'
    )


def button_updates(state: Connect4State):
    """Show one drop-button per real column; enable only legal ones."""
    legal = set(state.get_moves())
    ups = []
    for c in range(MAX_COLS):
        if c < state.cols:
            ups.append(gr.update(visible=True, interactive=(c in legal)))
        else:
            ups.append(gr.update(visible=False))
    return ups


def render_all(sess: dict, depth: int, top_k: int):
    state = sess["state"]
    return (
        sess,
        render_board_html(state),
        status_text(sess),
        render_tree_html(sess, depth, top_k),
        *button_updates(state),
    )


# -- event handlers ---------------------------------------------------------
def on_new_game(rows, cols, connect, n_iter, C, human_first, depth, top_k):
    if CFG.az_mode:
        rows, cols, connect = CFG.rows, CFG.cols, CFG.connect
    state = Connect4State(int(rows), int(cols), int(connect))
    sess = {
        "state": state,
        "root": None,
        "root_state": None,
        "human": P1 if human_first else P2,
    }
    if state.to_play != sess["human"]:  # computer moves first
        computer_think(sess, n_iter, C)
    return render_all(sess, depth, top_k)


def on_drop(col, sess, n_iter, C, depth, top_k, reuse):
    state = sess["state"]
    if state.is_terminal() or col not in state.get_moves():
        return render_all(sess, depth, top_k)

    # Warm-start only for classic UCT trees (AZ always searches from scratch).
    reuse_root = None
    if (not CFG.az_mode) and reuse and sess.get("root") is not None \
            and "last_computer_move" in sess:
        after_computer = az_child_by_move(sess["root"], sess["last_computer_move"])
        if after_computer is not None:
            reuse_root = az_child_by_move(after_computer, col)

    state.do_move(col)  # human move
    if not state.is_terminal():
        computer_think(sess, n_iter, C, reuse_root=reuse_root)  # computer replies
    return render_all(sess, depth, top_k)


def on_view_change(sess, depth, top_k):
    if not sess:
        return gr.update()
    return render_tree_html(sess, depth, top_k)


def build_ui() -> "gr.Blocks":
    if CFG.az_mode:
        title = "Connect-4 AlphaZero"
        heading = (
            "# Connect-4 vs. AlphaZero\n"
            f"Checkpoint: `{CFG.checkpoint_path}`"
            + (f" (iter {CFG.iteration})" if CFG.iteration is not None else "")
            + ". Search uses PUCT-MCTS guided by the network's policy and value "
            "(no random rollouts). You are 🔴, the computer is 🟡 "
            "(unless you let it move first)."
        )
        search_heading = "### Search (AlphaZero MCTS)"
        c_label = "PUCT exploration constant c_puct"
        c_default = 1.5
        n_default = 200
        value_blurb = (
            "Each node shows its board, visit count **N** and backed-up state "
            "value **V** for the side to move ([-1, 1]). Edges are labeled with "
            "the column played; the thick green path is the most-visited "
            "(chosen) line."
        )
    else:
        title = "Connect-4 MCTS"
        heading = (
            "# Connect-4 vs. MCTS\n"
            "You are 🔴, the computer is 🟡 (unless you let it move first). "
            "Configure the board and search below, then drop discs. The search "
            "tree the computer builds each turn is shown at the bottom."
        )
        search_heading = "### Search (MCTS)"
        c_label = "Exploration constant C"
        c_default = 1.41
        n_default = 400
        value_blurb = (
            "Each node shows its board, visit count **N** and estimated state "
            "value **V** for the side to move ([0, 1]). Edges are labeled with "
            "the column played; the thick green path is the most-visited "
            "(chosen) line."
        )

    board_locked = CFG.az_mode

    with gr.Blocks(title=title) as demo:
        gr.Markdown(heading)
        sess_state = gr.State()

        with gr.Row():
            # ---- Settings column ----
            with gr.Column(scale=1):
                gr.Markdown("### Board")
                rows = gr.Slider(
                    4, MAX_ROWS, value=CFG.rows, step=1, label="Rows",
                    interactive=not board_locked,
                )
                cols = gr.Slider(
                    4, MAX_COLS, value=CFG.cols, step=1, label="Columns",
                    interactive=not board_locked,
                )
                connect = gr.Slider(
                    3, 5, value=CFG.connect, step=1, label="In-a-row to win",
                    interactive=not board_locked,
                )
                if board_locked:
                    gr.Markdown("*Board size is fixed by the checkpoint.*")
                human_first = gr.Checkbox(value=True, label="You move first")

                gr.Markdown(search_heading)
                n_iter = gr.Slider(10, 5000, value=n_default, step=10,
                                   label="Thinking budget N (simulations)")
                C = gr.Slider(0.0, 3.0, value=c_default, step=0.01,
                              label=c_label)
                reuse = gr.Checkbox(
                    value=False,
                    label="Reuse tree across turns (warm start)",
                    interactive=not CFG.az_mode,
                    visible=not CFG.az_mode,
                )

                gr.Markdown("### Tree view")
                depth = gr.Slider(1, 6, value=3, step=1, label="Max display depth")
                top_k = gr.Slider(1, 7, value=4, step=1,
                                  label="Top-k children per node")

                new_game_btn = gr.Button("New game", variant="primary")
                gr.Markdown(
                    "*Board/win settings apply on **New game**. N, C and the tree "
                    "view update immediately.*"
                )

            # ---- Play column ----
            with gr.Column(scale=2):
                status = gr.Markdown("### Start a new game.")
                col_buttons = []
                with gr.Row():
                    for c in range(MAX_COLS):
                        col_buttons.append(
                            gr.Button(f"▼ {c}", visible=False, min_width=40)
                        )
                board = gr.HTML()

        gr.Markdown("## Search tree")
        gr.Markdown(value_blurb)
        tree = gr.HTML()

        outputs = [sess_state, board, status, tree, *col_buttons]

        # Wire events.
        new_game_btn.click(
            on_new_game,
            inputs=[rows, cols, connect, n_iter, C, human_first, depth, top_k],
            outputs=outputs,
        )
        for c, btn in enumerate(col_buttons):
            btn.click(
                partial(on_drop, c),
                inputs=[sess_state, n_iter, C, depth, top_k, reuse],
                outputs=outputs,
            )
        for view_ctrl in (depth, top_k):
            view_ctrl.change(on_view_change, inputs=[sess_state, depth, top_k],
                             outputs=tree)

        # Start a game immediately on load.
        demo.load(
            on_new_game,
            inputs=[rows, cols, connect, n_iter, C, human_first, depth, top_k],
            outputs=outputs,
        )

    return demo


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Play Connect-4 against classic MCTS or an AlphaZero checkpoint",
    )
    p.add_argument(
        "checkpoint",
        nargs="?",
        default=None,
        help="Optional AlphaZero checkpoint (.pt). If set, use PUCT-MCTS.",
    )
    p.add_argument(
        "--checkpoint",
        dest="checkpoint_opt",
        default=None,
        help="Same as the positional checkpoint argument",
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    global CFG
    args = parse_args(argv)
    path = args.checkpoint_opt or args.checkpoint
    if path:
        device = torch.device(args.device)
        net, meta = load_az_checkpoint(path, device)
        CFG = PlayConfig(
            az_mode=True,
            net=net,
            device=device,
            rows=meta["rows"],
            cols=meta["cols"],
            connect=meta["connect"],
            checkpoint_path=path,
            iteration=meta.get("iteration"),
        )
        print(
            f"AlphaZero mode: {path}  board={CFG.rows}x{CFG.cols}  "
            f"device={device}  iter={CFG.iteration}"
        )
    else:
        CFG = PlayConfig()
        print("Classic MCTS mode (pass --checkpoint PATH to use AlphaZero)")

    build_ui().launch()


if __name__ == "__main__":
    main()
