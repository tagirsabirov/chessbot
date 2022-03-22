import chess
from evaluation import Evaluate
from tensorflow import keras
import numpy as np
model = keras.models.load_model('NN-model-dense.h5')

squares_index = {
  'a': 0,
  'b': 1,
  'c': 2,
  'd': 3,
  'e': 4,
  'f': 5,
  'g': 6,
  'h': 7
}

def square_to_index(square):
  letter = chess.square_name(square)
  return 8 - int(letter[1]), squares_index[letter[0]]

def split_dims(board):
    # this is the 3d matrix
    board3d = np.zeros((14, 8, 8), dtype=np.int8)

    # here we add the pieces's view on the matrix
    for piece in chess.PIECE_TYPES:
      for square in board.pieces(piece, chess.WHITE):
        idx = np.unravel_index(square, (8, 8))
        board3d[piece - 1][7 - idx[0]][idx[1]] = 1
      for square in board.pieces(piece, chess.BLACK):
        idx = np.unravel_index(square, (8, 8))
        board3d[piece + 5][7 - idx[0]][idx[1]] = 1

    # add attacks and valid moves too
    # so the network knows what is being attacked
    aux = board.turn
    board.turn = chess.WHITE
    for move in board.legal_moves:
        i, j = square_to_index(move.to_square)
        board3d[12][i][j] = 1
    board.turn = chess.BLACK
    for move in board.legal_moves:
        i, j = square_to_index(move.to_square)
        board3d[13][i][j] = 1
    board.turn = aux

    return board3d

def neural_eval(board):
    board3d = split_dims(board)
    board3d = np.expand_dims(board3d, 0)
    board3d = np.asarray(board3d).astype(np.float32)
    return model.predict(board3d)[0][0]


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
            return neural_eval(board)
            
        if maximizing_player:
            #value is the board value
            max_value = float("-inf")
            for move in board.legal_moves:
                # Calculate all possible moves
                board.push(move)
                # Calculate a depth further recursively
                #choosing the largest value to maximize
                value = chessbot.minimax(board, depth - 1, False, alpha, beta)
                board.pop()
                max_value = max(max_value, value)
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            return max_value
        #If it's not your move
        else:
            min_value = float("inf")
            for move in board.legal_moves:
                board.push(move)
                value = chessbot.minimax(board, depth - 1, True, alpha, beta)
                board.pop()
                min_value = min(min_value, value)
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return min_value

    def move(board, depth, maximizing_player, alpha, beta):
        if maximizing_player:
            best_move = None
            max_value = float("-inf")
            for move in board.legal_moves:
                board.push(move)
                value = chessbot.minimax(board, depth - 1, True, alpha, beta)
                board.pop()
                if value > max_value:
                    max_value = value
                    best_move = move
            return best_move, max_value
        else:
            best_move = None
            min_value = float("inf")
            for move in board.legal_moves:
                board.push(move)
                value = chessbot.minimax(board, depth - 1, False, alpha, beta)
                board.pop()
                if value < min_value:
                    min_value = value
                    best_move = move
            return best_move, min_value
        