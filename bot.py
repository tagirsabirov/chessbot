import chess
from evaluation import Evaluate


class chessbot:
    def minimax(board, depth, maximizing_player, alpha, beta):
        '''Recursive algorithm to search a certain depth and subtract 1 for every depth counted.
        Calculate all moves to a depth first, and then minimize and maximize going up.'''

        if depth == 0 or board.is_game_over():
            #Only calculate the evaluation once per node, at the lowest depth of the tree
            # return Evaluate.evaluate_board(board), None
            return Evaluate.evaluate_board(board);
            
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
        