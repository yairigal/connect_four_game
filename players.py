"""Connect found players logic module."""
import random

import math

from game import *


class Player:
    """Connect four player logic implementation class.

    This player evaluates each spot and picks the spot with the most reward given to him.
    To evaluates the spots the player uses Negamax (prettier way to code Minimax) with alpha-beta pruning.
    The spots at the max depth are evaluated by a heuristic.
    The heuristic is documented at calculate_reward function.

    Attributes:
        mark (str): the icon of the player on the board.
        opponent (str): the icon of the opponent on the board.
        game (ConnectFour): the game instance.
        difficulty (number): the max depth of the minimax tree.
    """

    def __init__(self, mark: str, opponent: str, game: ConnectFour, difficulty=5, **kwargs):
        self.mark = mark
        self.opponent = opponent
        self.game = game
        self.depth = difficulty

    @property
    def won(self):
        """If the player won."""
        return self.game.is_winner(self.mark)

    @property
    def lost(self):
        """If the opponent won."""
        return self.game.is_winner(self.opponent)

    @contextmanager
    def placed_at(self, index):
        """Simulate a turn.

        Place of the selected column, and remove it.

        Args:
            index (number): the index of the column so place.
         """
        self.game.place(self.mark, index)
        yield
        self.game.remove()

    def turn(self):
        """Play a turn in the game.

        Returns:
            number. the index of the column to place at.
        """
        _, spot = self.negamax(depth=self.depth)
        return spot

    def negamax(self, depth=5, alpha=-math.inf, beta=math.inf):
        """Implementation of the Negamax search algorithm.

        Args:
            depth (number): the depth of the search tree.
            alpha (number): the min value the max player is looking for.
            beta (number): the max value the min player is looking for.

        Returns:
            tuple. the heuristic value of the option picked, the index of the column to place at.
        """
        opponent = self.__class__(self.opponent, self.mark, self.game)

        if depth == 0:
            return self.calculate_reward(), -1

        value = -math.inf
        best_move = -1
        columns = [column for column in range(self.game.width) if not self.game.is_column_full(column)]
        if len(columns) == 1:
            return 500_000, columns[0]

        random.shuffle(columns)
        for index in columns:
            with self.placed_at(index):
                if self.won:
                    return 100_000 * depth, index

                if self.lost:  # impossible if, here for safety.
                    return -500_000 * depth, index

                reward, _ = opponent.negamax(depth - 1, alpha=-beta, beta=-alpha)

                # alpha beta pruning
                if -reward > value:
                    value = -reward
                    best_move = index

                alpha = max(alpha, value)
                if alpha >= beta:
                    break

        return value, best_move

    def calculate_reward(self):
        """Evaluate a heuristic value for that current game snapshot.

        The heuristic is giving points to the player based on how much is he close to winning or losing.
        If the player has won it wins 100k points.
        if he lost it loses 500k points (=-500k) since losing is much worse than winning, and it should be his
        highest priority to not lose.

        If the game is not at a losing or winning state, the heuristic calculates how much the player and opponent
        is close to winning.
        the function checks if the player has 3 spots in a row with an empty spot that can be completed to a win.
        each of that sequence is counted as 12 points.
        Same goes for the opponent, it gets 20 (=-20 for the player) points for each sequence it has with 3 spots and
        an empty spot.


        Returns:
            number. the heuristic value.
        """
        if self.won:
            return 100_000

        if self.lost:
            return -500_000

        reward = 0

        # if losing (=enemy has 3 in a row with potential place to put) -> minus points
        reward += -20 * self.game.possible_win(self.opponent, n=3)

        reward += 12 * self.game.possible_win(self.mark, n=3)

        return reward


class RandomPlayer:
    def __init__(self, mark: str, opponent: str, game: ConnectFour, **kwargs):
        self.mark = mark
        self.opponent = opponent
        self.game = game

    def turn(self):
        optional_columns = list(range(0, self.game.width))
        for column in optional_columns[:]:
            if self.game.is_column_full(column):
                optional_columns.remove(column)

        return random.choice(optional_columns)


class HumanPlayer:
    def __init__(self, mark: str, opponent: str, game: ConnectFour, **kwargs):
        self.mark = mark
        self.opponent = opponent
        self.game = game

    def turn(self):
        col = input(f'Enter column between 1 to {self.game.width}: ')
        return int(col) - 1
