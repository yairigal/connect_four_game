"""Connect found players logic module."""
import copy
from functools import partial
import logging
import multiprocessing
import pathlib
import pickle
import random

import math

from game import *

logging.basicConfig(level=logging.DEBUG,
                    datefmt='%Y-%m-%d %H:%M:%S',
                    format='%(asctime)s.%(msecs)03d: %(message)s')
logger = logging.getLogger(__name__)

NUM_CORES = 4

WEIGHTS = {
    'winning': 18.25,  # old=12
    'losing': -10  # old=-20
}


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


PLAYERS = {
    'computer': MinimaxPlayer,
    'random': RandomPlayer,
    'human': HumanPlayer,
    'monte_carlo': MonteCarloPlayer,
}

WORKER_POOL = multiprocessing.Pool(processes=NUM_CORES)
