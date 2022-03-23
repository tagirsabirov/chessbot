import chess
import chess.engine
import chess.pgn
from bot import chessbot

board = chess.Board()

game = chess.pgn.Game()
game.headers["Event"] = "Chessbot vs Stockfish"
# node = game.add_variation(chess.Move.from_uci("e2e4"))
# node = node.add_variation(chess.Move.from_uci("e7e5"))
# node.comment = "Comment"

with chess.engine.SimpleEngine.popen_uci(r"stockfish\stockfish_14.1_win_x64_avx2.exe") as engine:
  while True:
    move = chessbot.move(board, 1, True, float("-inf"), float("inf"))[0]
    board.push(move)
    print(chess.Move.from_uci(move))
    game.add_variation(chess.Move.from_uci(str(move)))
    if board.is_game_over():
      print(game)
      break

    move = engine.analyse(board, chess.engine.Limit(time=1), info=chess.engine.INFO_PV)['pv'][0]
    board.push(move)
    game.add_variation(chess.Move.from_uci(str(move)))
    if board.is_game_over():
      print(game)
      break