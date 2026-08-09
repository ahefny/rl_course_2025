"""Connect-4 AlphaZero package.

Layout:
    core.py            – game rules + classic UCT-MCTS
    train.py           – AlphaZero network, self-play, serial trainer
    train_parallel.py  – batched parallel self-play trainer
    play.py            – Gradio play UI
    viz.py             – Gradio policy/value visualizer

Run from the repo root, e.g.:
    python model_based_rl/connect4_alphazero/train.py --help
"""
