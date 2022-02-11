from pickle import FALSE
import chess
import chess.svg
from pkg_resources import evaluate_marker
from evaluation import Evaluate

# create board object
board = chess.Board()
legal_moves = list(board.legal_moves)


def minimax(board, depth, maximizing_player):
    '''Recursive algorithm to search a certain depth and subtract 1 for every depth counted.
    Calculate all moves to a depth first, and then minimize and maximize going up.'''

    if depth == 0 or board.is_game_over():
        #Only calculate the evaluation once per node, at the lowest depth of the tree
        return Evaluate.evaluate_board(board)
         
    if maximizing_player:
        #value is the board value
        value = -float("inf")
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
        for move in board.legal_moves:
            board.push(move)
            value = min(value, minimax(board, depth - 1, True))
            board.pop()
        return value

board.push_san("e4")
board.push_san("e5")
board.push_san("Nf3")
board.push_san("Qg5")


print("Board evaluation: " + str(Evaluate.evaluate_board(board)))

print("Minimax evaluation: " + str(minimax(board, 3, False)))

