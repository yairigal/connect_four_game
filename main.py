import argparse
import json
import logging
from pathlib import Path

from game import ConnectFour
import players

logging.basicConfig(level=logging.DEBUG,
                    datefmt='%Y-%m-%d %H:%M:%S',
                    format='%(asctime)s.%(msecs)03d: %(message)s')
logger = logging.getLogger(__name__)


def player_vs_random_statistics(n=100):
    """Run n games with player vs random player."""
    wins = [0, 0]
    draws = 0
    for i in range(n):
        winner = ConnectFour.play(players.MinimaxPlayer, players.RandomPlayer, verbose=False, difficulty=5)
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
        winner = ConnectFour.play(players.MonteCarloPlayer, players.MinimaxPlayer, verbose=False, difficulty=difficulty,
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
        winner = ConnectFour.play(players.MonteCarloPlayer, players.RandomPlayer, verbose=False)
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
        winner = ConnectFour.play(players.MinimaxPlayer, players.MinimaxPlayer, verbose=False, difficulty=6)
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
                        help='one of: computer, human, random.')
    parser.add_argument('--player2',
                        type=str,
                        dest='player2',
                        default='human',
                        help='one of: computer, human, random.')
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


if __name__ == '__main__':
    args = parse_cli_arguments()
    p1 = players.PLAYERS[args.player1]
    p2 = players.PLAYERS[args.player2]
    ConnectFour.play(p1, p2, difficulty=args.difficulty, number_of_games=args.games)
    # player_vs_random_statistics(n=10000)
