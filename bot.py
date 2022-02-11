from pickle import FALSE
import chess
import chess.svg
from pkg_resources import evaluate_marker
from evaluation import Evaluate
import math

# create board object
board = chess.Board()
legal_moves = list(board.legal_moves)


def minimax(board, depth, maximizing_player):
    '''Recursive algorithm to search a certain depth and subtract 1 for every depth counted.
    Calculate all moves to a depth first, and then minimize and maximize going up.'''

    if depth == 0 or board.is_game_over():
        #Only calculate the evaluation once per node, at the lowest depth of the tree
        return Evaluate.evaluate_board(board), None
         
    if maximizing_player:
        #value is the board value
        max_value = float("-inf")
        for move in board.legal_moves:
            # Calculate all possible moves
            board.push(move)
            # Calculate a depth further recursively
            #choosing the largest value to maximize
            value = minimax(board, depth - 1, False)[0]
            board.pop()
            if value > max_value:
                max_value = value
                best_move = move
        return max_value, best_move
    #If it's not your move
    else:
        min_value = float("inf")
        for move in board.legal_moves:
            board.push(move)
            value = minimax(board, depth - 1, True)[0]
            board.pop()
            if value < min_value:
                min_value = value
                best_move = move
        return min_value, best_move

board.push_san("e4")
board.push_san("e5")
board.push_san("Nf3")
board.push_san("Qg5")
board.push_san("Nxg5")


print("Board evaluation: " + str(Evaluate.evaluate_board(board)))

print("Minimax evaluation: " + str(minimax(board, 2, False)))

