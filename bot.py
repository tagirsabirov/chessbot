# uses chess python library
from pickle import FALSE
import chess
import chess.svg
from IPython.display import SVG
from evaluation import evaluate

# create board object
board = chess.Board()
legal_moves = list(board.legal_moves)
SVG(chess.svg.board(board=board,size=400))

# def evaluate(board):
#     #piece counting



# recursive algorithm to search a certain depth and subtract 1 for every depth counted.
# Calculate all moves to a depth first, and then minimize and maximize going up.
def minimax(board, depth, maximizing_player):
    if depth == 0 or board.is_game_over():
        return evaluate.evaluate_board()
        #return board evaluation value
        #evaluate(board)
    if maximizing_player == True:
        #value is the board value
        value = -float("inf")
        # value = -evaluate.evaluate_board()
        for move in board.legal_moves:
            # Calculate all possible moves
            board.push(move)
            # Calculate a depth further recursively
            #choosing the largest value to maximize 
            value = max(value, minimax(board, depth - 1, False))
            board.pop()
        return value
    #If it's not your move
    else:
        value = float("inf")
        # value = evaluate.evaluate_board()
        for move in board.legal_moves:
            board.push(move)
            #choosing the smallest value to minimize
            value = min(value, minimax(board, depth - 1, True))
            board.pop()
        return value

board.push_san("d4")
SVG(chess.svg.board(board=board,size=400))
print(minimax(board, 5, FALSE))