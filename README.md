# Instinct Chess Bot

A transformer powered chess bot that uses "instinct" rather than search to find
the best move. Based on Google DeepMind's paper
[Grandmaster-Level Chess Without Search](https://arxiv.org/html/2402.04494v1).

## Installation

```
pip install git+https://github.com/jakesmd/instinct-chess-bot.git
```

## Usage

```python
import chess, chess.engine

instinct = chess.engine.SimpleEngine.popen_uci("instinct-chess-bot")

board = chess.Board()
while not board.is_game_over():
    result = instinct.play(board, chess.engine.Limit(time=1))
    board.push(result.move)

print(board.outcome())

instinct.quit()
```
