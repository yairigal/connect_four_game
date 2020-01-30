import argparse
from collections import defaultdict
from contextlib import contextmanager
import copy
from functools import partial
from io import StringIO
import itertools
import json
import logging
import multiprocessing
import pathlib
from pathlib import Path
import pickle
import random
import sys

import math

logging.basicConfig(level=logging.DEBUG,
                    datefmt='%Y-%m-%d %H:%M:%S',
                    format='%(asctime)s.%(msecs)03d: %(message)s')
logger = logging.getLogger(__name__)

NUM_CORES = 4

WEIGHTS = {
    'winning': 18.25,  # old=12
    'losing': -10  # old=-20
}


class bcolors:
    """Console colors data class."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @staticmethod
    def text(message, color):
        color_object = getattr(bcolors, color.upper())
        return f"{color_object}{message}{bcolors.ENDC}"


@contextmanager
def mock_stdout():
    """Capture stdout and process the text."""
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    yield sys.stdout
    sys.stdout = old_stdout


#############################
###### INFRASTRUCTURE #######
#############################
def color(func):
    """Color each player's character in different color."""

    def wrapper(self):
        with mock_stdout() as output:
            func(self)
            char_to_color = {}
            colors = ['fail', 'green']
            for i, mark in enumerate(self.moves.keys()):
                char_to_color[mark] = colors[i]

            output = output.getvalue()
            for mark, color in char_to_color.items():
                output = output.replace(mark, bcolors.text(mark, color))

        print(output)

    return wrapper


class ColumnFull(OverflowError):
    def __init__(self, index, *args, **kwargs):
        super().__init__(f"Column {index} is full")


class ConnectFour:
    """Connect four game implementation.

    Attributes:
        height (number): the height of the board.
        width (number): the width of the board.
        board (list): 2d array representing the game's board.
        moves (dict): history of moves per player's character.
        last_moves (list): list of the last moves executed by order.
    """

    def __init__(self, height=6, width=7):
        self.height = height
        self.width = width
        self.board = [[None for _ in range(width)] for _ in range(height)]
        self.moves = defaultdict(list)
        self.last_moves = []

    @color
    def draw(self):
        """Print the board."""
        row_number = '   '.join(str(item + 1) for item in range(self.width))
        print(f'  {row_number}  ')
        delimiter = '-' * (self.width * 4 + 1)
        for row in self.board:
            print(delimiter)
            row_formatted = ' | '.join(' ' if item is None else item for item in row)
            print(f'| {row_formatted} |')

        print(delimiter)

    def place(self, mark, index):
        """Place a character at the column.

        Args:
            mark (str): the character to place.
            index (number): the index of the column to place in.
        """
        x = self._top(index)
        if x < 0:
            raise ColumnFull(index)

        self.board[x][index] = mark
        self.moves[mark].append((x, index))
        self.last_moves.append(index)

    def is_winner(self, mark):
        """Check if `mark` has 4 continues appearances in a row, column or diagonal.

        Args:
            mark (str): the player's character.

        Returns:
            bool. whether if `mark` won.
        """
        winning_sequence = [mark] * 4

        for sequence in self.rows_columns_diagonals():
            if self.is_sublist(winning_sequence, sequence):
                return True

        return False

    def remove(self):
        """Remove the last move placed on the board.

        Returns:
            number. the index of column removed from.
        """
        index = self.last_moves.pop()
        x = self._top(index) + 1
        self.moves[self.board[x][index]].remove((x, index))
        self.board[x][index] = None
        return index

    def is_column_full(self, column):
        return self.board[0][column] is not None

    def row(self, index):
        """Return the row with that index."""
        return [self.board[x][y] for x, y in self._item_generator(index, 0, d_x=0, d_y=1)]

    def column(self, index):
        """Return the column with that index."""
        return [self.board[x][y] for x, y in self._item_generator(0, index, d_x=1, d_y=0)]

    def diagonal(self, index):
        """Return the items on the diagonal."""
        if index < self.width:
            x, y = 0, index

        else:
            x, y = index - self.width + 1, 0

        return [self.board[x][y] for x, y in self._item_generator(x, y, d_x=1, d_y=1)]

    def anti_diagonal(self, index):
        """Return the items on the non main diagonal."""
        if index < self.width:
            x, y = 0, index

        else:
            x, y = index - self.width + 1, self.width - 1

        return [self.board[x][y] for x, y in self._item_generator(x, y, d_x=1, d_y=-1)]

    def rows_columns_diagonals(self):
        """Return all the rows, columns and diagonals in the board.

        Yields:
            list. lists of the rows, columns and diagonals.
        """
        for diagonal_index in range(self.width + self.height - 1):
            yield self.diagonal(diagonal_index)
            yield self.anti_diagonal(diagonal_index)

        yield from (self.row(i) for i in range(self.height))
        yield from (self.column(i) for i in range(self.width))

    def possible_win(self, mark, n=3):
        """Count the amount of sequences that is close to win.

        Args:
            mark (str): the player's mark.
            n (number): the length of the mark sequence (the rest will be filled with None's).

        Returns:
            number. the amount of possible-wins sequences.
        """
        winning_sequence = [mark] * n + [None] * (4 - n)
        winning_permutations = list(set(itertools.permutations(winning_sequence, r=4)))

        amount = 0
        for sequence in self.rows_columns_diagonals():
            for permutation in winning_permutations:
                if self.is_sublist(permutation, sequence):
                    amount += 1

        return amount

    def __hash__(self):
        return hash(tuple(tuple(row) for row in self.board))

    @staticmethod
    def is_sublist(contained_list, container_list):
        """Validate if the contained list is contained in the container list.

        Args:
            contained_list (iterable): list of items.
            container_list (iterable): list of items.

        Returns:
            bool. if contained_list is in container_list in the exact order.
        """
        return str(contained_list)[1:-1] in str(container_list)

    def _top(self, column):
        """Returns the index of the lowest empty spot on that column.

        Args:
            column (number): the index of the column.

        Returns:
            number. the lowest empty spot index on that column.
        """
        columns = list(zip(*self.board))
        return self.height - len([item for item in columns[column] if item is not None]) - 1

    def _item_generator(self, x, y, d_x, d_y, direction=1):
        """Sequence item generator.

        The item generator yields the indices of items starting from x,y and continues with d_x and d_y.

        Args:
            x (number): index of the x-axis starting point.
            y (number): index of the y-axis starting point.
            d_x (number): the rate of the progress in the x-axis.
            d_y (number): the rate of the progress in the y-axis.
            direction (number): can be -1 or 1, indicates the direction of the x and y relative to d_x and d_y.

        Yields:
            tuple. the indices of the next spot.
        """
        d_x *= direction
        d_y *= direction
        while 0 <= x < self.height and 0 <= y < self.width:
            yield x, y
            x += d_x
            y += d_y

    @staticmethod
    def all_columns_are_full(board):
        return all(board.is_column_full(column) for column in range(board.width))

    @staticmethod
    def who_won(players, game):
        """Checks which one of the players won.

        Args:
            players (list): list of Player objects.
            game (ConnectFour): the game object.

        Returns:
            number. the index of the winning player in the list, None if no one won.
        """
        for index, player in enumerate(players):
            if game.is_winner(player.mark):
                return index

    @staticmethod
    def play(p1, p2, game=None, verbose=True, **players_args):
        """Start a game vs human player."""
        if game is None:
            game = ConnectFour()

        if type(p1) is not type and type(p2) is not type:
            players = [p1, p2]

        elif type(p1) is type and type(p2) is type:
            players = [
                p1('x', 'o', game, **players_args),
                p2('o', 'x', game, **players_args)
            ]

        else:
            raise TypeError('Both player arguments should be instances or types')

        turn = 0
        while True:
            index = ConnectFour.who_won(players, game)
            if index is not None:
                if verbose:
                    print(f'Player {index + 1} WON')

                return index

            if ConnectFour.all_columns_are_full(game):
                if verbose:
                    print('DRAW')

                return -1

            player = players[turn]
            game.place(player.mark, player.turn())
            if verbose:
                game.draw()

            turn = 1 - turn


#############################
######### PLAYERS ###########
#############################


def gradient_decent(epoch_size=100, change_rate=5):
    difficulty = 5
    games = 500
    current_win_percentage = 0
    while True:
        play_game = partial(ConnectFour.play, MinimaxPlayer, MonteCarloPlayer, verbose=False)
        wins = sum(play_game(number_of_games=games, difficulty=difficulty) == 0 for _ in range(epoch_size))

        win_percentage = wins / epoch_size

        logger.info(f'Minimax win rate={win_percentage}, '
                    f'losing_weight={WEIGHTS["winning"]}')

        if win_percentage > current_win_percentage:  # There is an improvement
            WEIGHTS['winning'] += change_rate

        else:  # There is no improvement
            change_rate = -change_rate / 2
            WEIGHTS['winning'] += change_rate

        current_win_percentage = win_percentage


class MinimaxPlayer:
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
        reward += WEIGHTS['losing'] * self.game.possible_win(self.opponent, n=3)

        reward += WEIGHTS['winning'] * self.game.possible_win(self.mark, n=3)

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


class MonteCarloPlayer:
    """Player that uses monte-carlo strategy to play.

    This player plays amount of games before each turn.
    The player plays against a random player for set amount of games (say 500) and picks the column which had the mos
    win when starting with it.

    Attributes:
        mark (str): the icon of the player on the board.
        opponent (str): the icon of the opponent on the board.
        game (ConnectFour): the game instance.
        number_of_games (int): the amount of games played each turn.

    """

    def __init__(self, mark: str, opponent: str, game: ConnectFour, number_of_games=500, **kwargs):
        self.mark = mark
        self.opponent = opponent
        self.game = game
        self.games = number_of_games

    def _play_game(self, opponent=RandomPlayer, my_turn=True):
        """Play one game against a random player.

        Args:
            opponent (Player): the opponent to play against.
            my_turn (bool): whether its my turn or not.

        Returns:
            bool. whether our agent won.
        """
        state = copy.deepcopy(self.game)
        player1 = RandomPlayer(self.mark, self.opponent, state)
        player2 = opponent(self.opponent, self.mark, state)
        if my_turn:
            winner = ConnectFour.play(p1=player1, p2=player2, game=state, verbose=False)

        else:
            winner = ConnectFour.play(p1=player2, p2=player1, game=state, verbose=False)

        if my_turn:
            return winner == 0

        else:
            return winner == 1

    def _pick_random_column(self):
        """Pick a random column of the remaining columns.

        Returns:
            number. the column selected.

        Raises:
            OverflowError: whether the board is full.
        """
        optional_columns = list(range(0, self.game.width))
        random.shuffle(optional_columns)
        for column in optional_columns:
            if not self.game.is_column_full(column):
                return column

        raise OverflowError('Game is full, no columns left')

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

    def _monte_carlo_estimation(self, games):
        """Play set amout of games to choose a column to play.

        Args:
            games (number): the amount of games to play.

        Returns:
            list. list containing the amount of wins and losses.
        """
        winnings = [0] * self.game.width
        for i in range(self.game.width):
            if self.game.is_column_full(i):
                winnings[i] = -1

        # play 500 games against random player.
        for i in range(games):
            column = self._pick_random_column()
            with self.placed_at(column):
                if self._play_game(my_turn=False):
                    winnings[column] += 1

        return winnings

    def turn(self):
        """Play a turn.

        This function calls the monte-carlo estimation in parallel to speed up evaluation.

        Returns:
            number. the column to place at.
        """
        chunk = self.games // NUM_CORES
        workers_outputs = WORKER_POOL.map(self._monte_carlo_estimation, [chunk] * NUM_CORES)
        winnings = [0] * self.game.width
        for win in workers_outputs:
            for i in range(len(winnings)):
                winnings[i] += win[i]

        # pick the column with most wins.
        return winnings.index(max(winnings))


class TDLearningPlayer:
    """Player that learns to play using Temporal difference learning.

    This Player uses reinforcement-learning to learn how to play well.
    Each state is saved, and is being evaluated by the end state (win lost or draw).
    """
    SAVE_FILE = pathlib.Path('q_model.data')

    def __init__(self, mark: str, opponent: str, game: ConnectFour, **kwargs):
        self.mark = mark
        self.opponent = opponent
        self.game = game

        self.exp_rate = 0.3
        self.gamma = 0.7
        self.lr = 0.5
        self.states = self.load_model(self.SAVE_FILE)

    def turn(self):
        return self._pick_best_column()

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

    def _pick_random_column(self):
        """Pick a random column of the remaining columns.

        Returns:
            number. the column selected.

        Raises:
            OverflowError: whether the board is full.
        """
        optional_columns = list(range(0, self.game.width))
        random.shuffle(optional_columns)
        for column in optional_columns:
            if not self.game.is_column_full(column):
                return column

        raise OverflowError('Game is full, no columns left')

    def _pick_best_column(self):
        """Pick the best column possible stored.

        Search all the visited states and pick the action which results with the most value.

        Returns:
            number. the best column possible for this state.
        """
        possible_actions = (column for column in range(self.game.width) if not self.game.is_column_full(column))
        values_per_action = {}
        for action in possible_actions:
            values_per_action[action] = self.states[(hash(self.game), action)]

        best_reward = max(values_per_action.values())
        top_cols = [action for action, reward in values_per_action.items() if reward == best_reward]
        return random.choice(top_cols)

    def _get_action(self):
        """Get action based on the current state.

        This function gets the action based on the exploration-exploitation trade-off.

        Returns:
            number. the column that is being picked.
        """
        if random.random() < self.exp_rate:  # exploration
            return self._pick_random_column()

        else:  # exploitation
            return self._pick_best_column()

    def feed_rewards(self, played_states, reward, action):
        """Back propogate reward back to all visited states.

        Args:
            played_states (list): states played this game.
            reward (number): the reward for the game goal.
            action (number): the last action for reach this state.
        """
        for state in reversed(played_states):
            self.states[(state, action)] += self.lr * (self.gamma * reward - self.states[(state, action)])
            reward = self.states[(state, action)]

    def _game_ends(self, players):
        """Check if the game ends

        Args:
            players (list): list of the players

        Returns:
            number. 1 if the the first player won, 0 if the first player lost, 0.5 for draw and -1 if the game didn't
            end.
        """
        index = ConnectFour.who_won(players, self.game)
        if index is not None:
            return 1 - index

        elif ConnectFour.all_columns_are_full(self.game):
            return 0.5

        return -1

    def train(self, player2=None):
        """Train the player against other players

        Each 1000 iterations the model is being saved to a file.

        Args:
            player2 (Player): the player to train against, if None random player is picked.
        """
        run = 0
        while True:
            run += 1

            turn = 0
            game = ConnectFour()

            self.game = game
            p2 = player2
            if player2 is None:
                p2 = random.choice([MinimaxPlayer])

            players = [
                self,
                p2(self.opponent, self.mark, game, difficulty=4)
            ]

            # simulation
            played_states = []
            while True:
                player = players[turn]

                game_status = self._game_ends(players)
                if game_status >= 0:  # Game ends
                    if turn == 0:  # this agent's turn
                        # update his scores
                        self.feed_rewards(played_states, game_status, action)

                    break

                if turn != 0:  # the other agent
                    game.place(player.mark, player.turn())

                else:  # the agent turn
                    action = player._get_action()
                    game.place(player.mark, action)
                    played_states.append(hash(self.game))

                turn = 1 - turn  # Switch turns

            if run % 1000 == 0:
                logger.info('Saving model')
                self.save_model(self.states)

    @staticmethod
    def load_model(filename):
        if not filename.exists():
            return defaultdict(int)

        with filename.open('rb') as f:
            return pickle.load(f)

    @staticmethod
    def save_model(model):
        with TDLearningPlayer.SAVE_FILE.open('wb') as f:
            pickle.dump(model, f)


#############################
########## UTILS ############
#############################


def player_vs_random_statistics(n=100):
    """Run n games with player vs random player."""
    wins = [0, 0]
    draws = 0
    for i in range(n):
        winner = ConnectFour.play(MinimaxPlayer, RandomPlayer, verbose=False, difficulty=5)
        if winner == -1:
            draws += 1

        else:
            wins[winner] += 1

    logger.info(f'Negamax wins={wins[0]} random player wins={wins[1]} draws={draws}')


def monte_carlo_vs_negamax_statistics(n=100, difficulty=5, games=500):
    filename = Path(f'monte_carlo_minimax_diff{difficulty}_games{games}')
    if filename.exists():
        with filename.open('r') as f:
            wins = json.load(f)

    else:
        wins = [0, 0]

    for i in range(n):
        # print(f'Game #{i}')
        winner = ConnectFour.play(MonteCarloPlayer, MinimaxPlayer, verbose=False, difficulty=difficulty,
                                  games=games)
        wins[winner] += 1
        with filename.open('w') as f:
            json.dump(wins, f)

    print(f"MonteCarlo won {wins[0]} times while Minimax won {wins[1]} times")


def monte_carlo_vs_random_statistics(n=100):
    wins = [0, 0]
    draws = 0
    for i in range(n):
        print(f'Game #{i}')
        winner = ConnectFour.play(MonteCarloPlayer, RandomPlayer, verbose=False)
        if winner == -1:
            draws += 1

        else:
            wins[winner] += 1

    print(f"MonteCarlo won {wins[0]} times while Random won {wins[1]} times")


def player_vs_player_statistics(n=100):
    """Run n games with player vs itself."""
    wins = [0, 0]
    draws = 0
    for i in range(n):
        winner = ConnectFour.play(MinimaxPlayer, MinimaxPlayer, verbose=False, difficulty=6)
        if winner == -1:
            draws += 1

        else:
            wins[winner] += 1

    logger.info(f'Negamax wins={wins[0]} random player wins={wins[1]} draws={draws}')


def parse_cli_arguments():
    parser = argparse.ArgumentParser(description='Play connect four.')
    parser.add_argument('--player1',
                        type=str,
                        dest='player1',
                        default='computer',
                        help='one of: computer, human, random, monte_carlo')
    parser.add_argument('--player2',
                        type=str,
                        dest='player2',
                        default='human',
                        help='one of: computer, human, random, monte_carlo')
    parser.add_argument('--difficulty', dest='difficulty',
                        type=int,
                        default=5,
                        help='the computer player difficulty, notice that difficulty 6 and above takes a lot of time '
                             'to evaluate.')
    parser.add_argument('--games', dest='games',
                        type=int,
                        default=500,
                        help='the amount of games per turn that the monte_carlo agents will play.')
    args = parser.parse_args()
    return args


PLAYERS = {
    'computer': MinimaxPlayer,
    'random': RandomPlayer,
    'human': HumanPlayer,
    'monte_carlo': MonteCarloPlayer,
}

WORKER_POOL = multiprocessing.Pool(processes=NUM_CORES)

if __name__ == '__main__':
    args = parse_cli_arguments()
    p1 = PLAYERS[args.player1]
    p2 = PLAYERS[args.player2]
    ConnectFour.play(p1, p2, difficulty=args.difficulty, number_of_games=args.games)
    # player_vs_random_statistics(n=10000)
