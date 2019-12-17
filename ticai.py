from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
import numpy as np
import random


# Minimax: 1
# Random (Neural Net): 2

LEN_ITER = 5000  # Configure

WINNERS = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 4, 8], [2, 4, 6], [0, 3, 6], [1, 4, 7], [2, 5, 8]]
INITIAL = [0, 0, 0, 0, 0, 0, 0, 0, 0]


class Board:
    def __init__(self, board):
        self.board = board
        self.value = 0

    def is_over(self):
        if self.board.count(1) + self.board.count(2) > 8:
            return True

        for x in WINNERS:
            if self.board[x[0]] == 1 and self.board[x[1]] == 1 and self.board[x[2]] == 1:
                return True

            if self.board[x[0]] == 2 and self.board[x[1]] == 2 and self.board[x[2]] == 2:
                return True

        return False

    def is_winner(self):
        for x in WINNERS:
            if self.board[x[0]] == 1 and self.board[x[1]] == 1 and self.board[x[2]] == 1:
                return 1

            if self.board[x[0]] == 2 and self.board[x[1]] == 2 and self.board[x[2]] == 2:
                return -1

        return 0


def get_valid_moves(board):
    moves = []

    for i, x in enumerate(board):
        if x == 0:
            moves.append(i)

    return moves


def place_move(board, move, player):
    new_board = board[:]

    new_board[move] = player

    return Board(new_board)


def minimax(board, maximizing, depth):
    if board.is_over():
        return board.is_winner()

    mover = []
    if maximizing:
        value = -100  # Since -1, 1, 0: equivalent to -inf
        for move in get_valid_moves(board.board):
            node = place_move(board.board, move, 1)

            value = max(value, minimax(node, False, depth + 1))

            if depth == 0:      # Just get the first moves..
                mover.append((move, value))

        if depth == 0:  # Just get the first max move..
            return max(mover, key=lambda x: x[1])

        return value

    else:
        value = 100  # Since -1, 1, 0: equivalent to +inf
        for move in get_valid_moves(board.board):
            node = place_move(board.board, move, 2)

            value = min(value, minimax(node, True, depth + 1))

        return value


def board_print(board):
    print("\n")
    for x in range(3):
        print(board[x * 3: (x + 1) * 3])


def create_model(model_):
    model_.add(Dense(units=9, activation='relu'))
    model_.add(Dense(units=50, activation='relu'))
    model_.add(Dense(units=40, activation='relu'))
    model_.add(Dense(units=30, activation='relu'))
    model_.add(Dense(units=9, activation='softmax'))

    model_.compile(loss='categorical_crossentropy', optimizer='adam')

    return model_


def play(board, model_):
    while not board.is_over():
        # board_print(board.board)   # Turn on for board print

        move = -1
        while move not in get_valid_moves(board.board):     # Generate random valid move, possible dangerous with rand..
            move = random.randint(0, 8)

        root = place_move(board.board, move, 2)

        if root.is_over():
            break

        move_pc = minimax(root, True, 0)[0]     # Run Minimax and return the move for 1

        # Train the network
        model_.fit(np.array([root.board]), to_categorical([move_pc], num_classes=9), epochs=50, verbose=0)

        board = place_move(root.board, move_pc, 1)


model = create_model(Sequential())      # CNN probably better...

for i in range(LEN_ITER):   # Train the network
    print(i)
    root_ = Board(INITIAL)
    play(root_, model)
