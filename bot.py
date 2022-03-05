import chess
from evaluation import Evaluate
from tensorflow import keras
import numpy as np
model = keras.models.load_model('NNUE-model.h5')

def split_dims(board):
  # this is the 3d matrix
  board3d = np.zeros((14, 8, 8), dtype=np.int8)

# create board object
board = chess.Board()
legal_moves = list(board.legal_moves)

class chessbot:
    def minimax(board, depth, maximizing_player, alpha, beta):
        '''Recursive algorithm to search a certain depth and subtract 1 for every depth counted.
        Calculate all moves to a depth first, and then minimize and maximize going up.'''

        if depth == 0 or board.is_game_over():
            #Only calculate the evaluation once per node, at the lowest depth of the tree
            # return Evaluate.evaluate_board(board), None
            board3d = split_dims(board)
            board3d = np.expand_dims(board3d, 0)
            board3d = np.asarray(board3d).astype(np.float32)
            return model.predict(board3d)[0][0]
        if maximizing_player:
            #value is the board value
            max_value = float("-inf")
            for move in board.legal_moves:
                # Calculate all possible moves
                board.push(move)
                # Calculate a depth further recursively
                #choosing the largest value to maximize
                value = chessbot.minimax(board, depth - 1, False, alpha, beta)[0]
                board.pop()
                if value > max_value:
                    max_value = value
                    best_move = move
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            return max_value, best_move
        #If it's not your move
        else:
            min_value = float("inf")
            for move in board.legal_moves:
                board.push(move)
                value = chessbot.minimax(board, depth - 1, True, alpha, beta)[0]
                board.pop()
                if value < min_value:
                    min_value = value
                    best_move = move
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return min_value, best_move