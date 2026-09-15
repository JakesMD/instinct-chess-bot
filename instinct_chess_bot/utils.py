import chess, torch


_STATE_CHARS = sorted(set(".-12345678abcdefghPNBRQKpnbrqkwb"))
_TOKEN_BY_CHAR = {c: i for i, c in enumerate(_STATE_CHARS)}
_QUEEN_DIRS = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
_KNIGHT_DELTAS = [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]
_PROMO_PIECES = (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)

CONTEXT_SIZE = 72  # 71 state chars (board + side + castling + ep) + 1 move token
NUM_STATE_TOKENS = len(_STATE_CHARS)

def _build_action_list() -> list[str]:
    # Returns a sorted list of all the possible UCI moves.
    ucis = set()
    for from_sq in range(64):
        from_rank, from_file = chess.square_rank(from_sq), chess.square_file(from_sq)
        for rank_delta, file_delta in _QUEEN_DIRS:
            for distance in range(1, 8):
                to_rank, to_file = from_rank + rank_delta * distance, from_file + file_delta * distance
                if 0 <= to_rank < 8 and 0 <= to_file < 8:
                    ucis.add(chess.Move(from_sq, chess.square(to_file, to_rank)).uci())

        for rank_delta, file_delta in _KNIGHT_DELTAS:
            to_rank, to_file = from_rank + rank_delta, from_file + file_delta
            if 0 <= to_rank < 8 and 0 <= to_file < 8:
                ucis.add(chess.Move(from_sq, chess.square(to_file, to_rank)).uci())

    for from_rank, to_rank in ((6, 7), (1, 0)):
        for from_file in range(8):
            for file_delta in (-1, 0, 1):
                to_file = from_file + file_delta
                if 0 <= to_file < 8:
                    from_sq = chess.square(from_file, from_rank)
                    to_sq = chess.square(to_file, to_rank)
                    for piece in _PROMO_PIECES:
                        ucis.add(chess.Move(from_sq, to_sq, promotion=piece).uci())

    return sorted(ucis)


ACTION_LIST = _build_action_list()

# State tokens already occupy ids 0 to NUM_STATE_TOKENS-1.
TOKEN_BY_ACTION = {u: NUM_STATE_TOKENS + i for i, u in enumerate(ACTION_LIST)}
NUM_ACTION_TOKENS = len(ACTION_LIST)
NUM_TOKENS = NUM_STATE_TOKENS + NUM_ACTION_TOKENS


def tokenize_sample(board: chess.Board, action: chess.Move) -> torch.Tensor:
    # Turns a FEN in to a character per square: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR -> rnbqkbnrpppppppp................................PPPPPPPPRNBQKBNR
    state = "".join("." * int(c) if c.isdigit() else c for c in board.board_fen() if c != "/")

    state += "w" if board.turn == chess.WHITE else "b"
    state += "K" if board.has_kingside_castling_rights(chess.WHITE) else "-"
    state += "Q" if board.has_queenside_castling_rights(chess.WHITE) else "-"
    state += "k" if board.has_kingside_castling_rights(chess.BLACK) else "-"
    state += "q" if board.has_queenside_castling_rights(chess.BLACK) else "-"
    state += chess.square_name(board.ep_square) if board.ep_square is not None else "-."

    ids = [_TOKEN_BY_CHAR[c] for c in state]
    ids.append(TOKEN_BY_ACTION[action.uci()])
    return torch.tensor(ids, dtype=torch.long)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"