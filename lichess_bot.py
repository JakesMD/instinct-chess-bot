import os
import chess
import chess.engine
from dotenv import load_dotenv

from lichess_bot_runner import LichessBot

load_dotenv()

with chess.engine.SimpleEngine.popen_uci("instinct-chess-bot") as engine:
    def make_move(board: chess.Board) -> chess.Move:
        result = engine.play(board, chess.engine.Limit(time=1.0))
        return result.move

    bot = LichessBot(token=os.environ["LICHESS_TOKEN"], make_move=make_move)
    bot.run()