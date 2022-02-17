import chess
from evaluation import Evaluate
from bot import chessbot
board = chess.Board()

def play():   
    while not board.is_game_over():
        try:
            board.push_san(input("Enter Move:"))
        except:
            print("Illegal Move")
        x = chessbot.minimax(board, 4, False, float("-inf"), float("inf"))
        print(x[1])
        board.push(x[1])
    else:
        print("Game Over")

play()