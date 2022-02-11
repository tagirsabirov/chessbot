from ..evaluation import Evaluate
import chess
board = chess.Board()

board.push_san("e4")
board.push_san("e5")
board.push_san("Qh5")
board.push_san("Nc6")
board.push_san("Bc4")
board.push_san("Nf6")
board.push_san("Qxf7")

print(board.is_checkmate())

print(board)

def test_evaluate_board():

    result = Evaluate().evaluate_board()
    print(result)
