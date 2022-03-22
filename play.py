import chess
from bot import chessbot
board = chess.Board()

def play():   
    while not board.is_game_over():
        try:
            board.push_san(input("Enter Move:"))
        except:
            print("Illegal Move")
        x = chessbot.move(board, 2, False, float("-inf"), float("inf"))
        print(x[0])
        board.push(x[0])
    else:
        print("Game Over")
play()