# UCI Protocol: https://backscattering.de/chess/uci/

import os, sys
import chess, torch

from instinct_chess_bot.model import InstinctTransformer
from instinct_chess_bot.utils import pick_device

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "model.pt")


def load_model(path: str, device: str) -> InstinctTransformer:
    model = InstinctTransformer().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def parse_position(board: chess.Board, tokens: list[str]) -> None:
    if tokens[0] == "startpos":
        board.reset()
        rest = tokens[1:]
    else:  # fen
        fen_end = tokens.index("moves") if "moves" in tokens else len(tokens)
        board.set_fen(" ".join(tokens[1:fen_end]))
        rest = tokens[fen_end:]

    if rest and rest[0] == "moves":
        for uci in rest[1:]:
            board.push(chess.Move.from_uci(uci))


def main() -> None:
    device = pick_device()
    model = None
    board = chess.Board()

    for line in sys.stdin:
        tokens = line.split()
        if not tokens:
            continue
        cmd = tokens[0]

        if cmd == "uci":
            print("id name InstinctChessBot")
            print("id author JakesMD")
            print("uciok", flush=True)
        elif cmd == "isready":
            if model is None:
                model = load_model(MODEL_PATH, device)
            print("readyok", flush=True)
        elif cmd == "ucinewgame":
            board.reset()
        elif cmd == "position":
            parse_position(board, tokens[1:])
        elif cmd == "go":
            if model is None:
                model = load_model(MODEL_PATH, device)
            move = model.pick_move(board, device)
            print(f"bestmove {move.uci()}", flush=True)
        elif cmd == "quit":
            break


if __name__ == "__main__":
    main()
