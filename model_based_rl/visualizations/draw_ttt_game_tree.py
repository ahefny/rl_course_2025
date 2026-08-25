"""Draw the complete Tic-Tac-Toe game tree from a board position.

Example:
    python draw_ttt_game_tree.py "x...o...." --output ttt_tree

The board string is read left-to-right, top-to-bottom.  ``.`` denotes an
empty square.  X always moves first.  Rendering the empty-board tree produces
hundreds of thousands of nodes, so expect that output to be large.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import graphviz


WINNING_LINES = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
)


def winner(board: str) -> str | None:
    """Return the winning marker, or ``None`` when the board has no winner."""
    for first, second, third in WINNING_LINES:
        marker = board[first]
        if marker != "." and marker == board[second] == board[third]:
            return marker
    return None


def validate_board(board: str) -> None:
    """Raise ValueError unless *board* is a reachable Tic-Tac-Toe position."""
    if len(board) != 9:
        raise ValueError("A board must contain exactly 9 characters.")
    if invalid_markers := set(board) - {"x", "o", "."}:
        raise ValueError(
            f"Board contains invalid marker(s): {''.join(sorted(invalid_markers))!r}. "
            "Use only 'x', 'o', and '.'."
        )

    x_count = board.count("x")
    o_count = board.count("o")
    if not (x_count == o_count or x_count == o_count + 1):
        raise ValueError("Invalid move counts: X moves first and players alternate.")

    winners = {
        board[line[0]]
        for line in WINNING_LINES
        if board[line[0]] != "."
        and board[line[0]] == board[line[1]] == board[line[2]]
    }
    if len(winners) > 1:
        raise ValueError("Both players cannot have winning lines.")
    if "x" in winners and x_count != o_count + 1:
        raise ValueError("An X win requires X to have made the last move.")
    if "o" in winners and x_count != o_count:
        raise ValueError("An O win requires O to have made the last move.")


def board_label(
    board: str,
    highlighted_cell: int | None = None,
    background_color: str | None = None,
) -> str:
    """Format a board as a 3-by-3 Graphviz HTML table.

    ``highlighted_cell`` is the square filled by the move leading to this
    state.  The root has no highlighted square.
    """
    def cell_html(index: int, cell: str) -> str:
        background = ' BGCOLOR="#FFF59D"' if index == highlighted_cell else ""
        marker = cell.upper() if cell != "." else "&#183;"
        return f'<TD WIDTH="20" HEIGHT="20"{background}>{marker}</TD>'

    rows = []
    for row in range(0, 9, 3):
        cells = "".join(
            cell_html(index, cell)
            for index, cell in enumerate(board[row : row + 3], start=row)
        )
        rows.append(f"<TR>{cells}</TR>")
    table_background = f' BGCOLOR="{background_color}"' if background_color else ""
    return (
        f'<<TABLE BORDER="1" CELLBORDER="1" CELLSPACING="0"'
        f' CELLPADDING="0"{table_background}>{"".join(rows)}</TABLE>>'
    )


def legal_moves(board: str) -> Iterable[int]:
    """Yield empty-square indices in row-major order."""
    return (index for index, marker in enumerate(board) if marker == ".")


def build_game_tree(initial_board: str) -> graphviz.Digraph:
    """Build a Graphviz diagram of every legal continuation of *initial_board*.

    States are intentionally not merged: two different move sequences occupy
    different nodes, which makes the diagram a game tree rather than a graph
    of unique board positions.
    """
    validate_board(initial_board)
    tree = graphviz.Digraph("tic_tac_toe")
    tree.attr(rankdir="TB", nodesep="0.15", ranksep="0.35", bgcolor="white")
    tree.attr("node", shape="plain", fontname="Courier", fontsize="14")
    node_count = 0

    def add_node(board: str, player: str, last_move: int | None = None) -> str:
        nonlocal node_count
        node_id = f"n{node_count}"
        node_count += 1

        result = winner(board)
        win_background = {"x": "#BBDEFB", "o": "#F8BBD0"}.get(result)
        tree.node(
            node_id,
            label=board_label(board, last_move, background_color=win_background),
        )

        if result is None and "." in board:
            for move in legal_moves(board):
                child_board = board[:move] + player + board[move + 1 :]
                child_id = add_node(
                    child_board, "o" if player == "x" else "x", last_move=move
                )
                tree.edge(node_id, child_id)
        return node_id

    next_player = "x" if initial_board.count("x") == initial_board.count("o") else "o"
    add_node(initial_board, next_player)
    return tree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the full Tic-Tac-Toe game tree from a board string."
    )
    parser.add_argument(
        "board",
        help="Nine characters containing x, o, and . (for example: x...o....).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("tic_tac_toe_tree"),
        help="Output path without an extension (default: tic_tac_toe_tree).",
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=("svg", "pdf", "png"),
        default="svg",
        help="Image format produced by Graphviz (default: svg).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    board = args.board.lower()
    try:
        tree = build_game_tree(board)
        output_path = tree.render(
            filename=args.output.name,
            directory=str(args.output.parent),
            format=args.format,
            cleanup=True,
        )
    except (ValueError, graphviz.ExecutableNotFound) as error:
        raise SystemExit(f"Error: {error}") from error
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
