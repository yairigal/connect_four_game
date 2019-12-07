import sys
import time

from game import ConnectFour
from players import Player, HumanPlayer, RandomPlayer

PLAYERS = {
    'computer': Player,
    'random': RandomPlayer,
    'human': HumanPlayer
}


def all_columns_are_full(board):
    return all(board.is_column_full(column) for column in range(board.width))


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


def play(p1, p2):
    """Start a game vs human player."""
    game = ConnectFour()
    players = [
        PLAYERS[p1.lower()]('x', 'o', game),
        PLAYERS[p2.lower()]('o', 'x', game)
    ]

    turn = 0
    while True:
        index = who_won(players, game)
        if index is not None:
            print(f'Player {index + 1} WON')
            break

        if all_columns_are_full(game):
            print('DRAW')
            break

        time.sleep(1)
        player = players[turn]
        game.place(player.mark, player.turn())
        game.draw()
        turn = 1 - turn


def player_vs_random_statistics(n=100):
    """Run n games with player vs random player."""
    wins = [0, 0]
    draws = 0
    for _ in range(n):
        game = ConnectFour()
        players = [
            Player('x', 'o', game),
            RandomPlayer('o', 'x', game)
        ]

        turn = 0
        while True:
            index = who_won(players, game)
            if index is not None:
                print(f'Player {index + 1} WON')
                wins[index] += 1
                break

            if all_columns_are_full(game):
                print('DRAW')
                draws += 1
                break

            player = players[turn]
            game.place(player.mark, player.turn())
            turn = 1 - turn

    print(f'Negamax wins={wins[0]} random player wins={wins[1]} draws={draws}')


def player_vs_player_statistics(n=100):
    """Run n games with player vs itself."""
    wins = [0, 0]
    draws = 0
    for _ in range(n):
        start = time.time()
        game = ConnectFour()
        players = [
            Player('x', 'o', game),
            Player('o', 'x', game)
        ]

        turn = 0
        while True:
            index = who_won(players, game)
            if index is not None:
                print(f'Player {index + 1} WON')
                wins[index] += 1
                break

            if all_columns_are_full(game):
                print(f'DRAW {time.time() - start}s')
                draws += 1
                break

            player = players[turn]
            game.place(player.mark, player.turn())
            turn = 1 - turn

    print(f'P1 wins={wins[0]} P2 wins={wins[1]} draws={draws}')


if __name__ == '__main__':
    player1 = 'computer'
    player2 = 'human'

    if len(sys.argv) > 1:
        player1, player2 = sys.argv[1:]

    play(player1, player2)
