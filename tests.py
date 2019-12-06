from game import ConnectFour
from players import Player


def test_winner_row():
    # Row
    """
    x x x x
    """
    c = ConnectFour()
    assert not c.is_winner('x')
    c.place('x', 0)
    c.place('x', 1)
    c.place('x', 2)
    c.place('x', 3)

    assert c.is_winner('x')


def test_winner_column():
    # Column
    """
    x
    x
    x
    x
    """
    c = ConnectFour()
    assert not c.is_winner('x')
    c.place('x', 0)
    c.place('x', 0)
    c.place('x', 0)
    c.place('x', 0)

    assert c.is_winner('x')


def test_winner_diagonal():
    # diag
    """
    x
    o x
    o o x
    o o o x
    """
    c = ConnectFour()
    assert not c.is_winner('x')
    c.place('o', 0)
    c.place('o', 0)
    c.place('o', 0)
    c.place('o', 1)
    c.place('o', 1)
    c.place('o', 2)
    c.place('x', 0)
    c.place('x', 1)
    c.place('x', 3)
    c.place('x', 2)

    assert c.is_winner('x')


def test_remove():
    c = ConnectFour()
    assert not c.is_winner('x')
    c.place('x', 0)
    c.place('x', 1)
    c.place('x', 2)
    c.place('x', 3)
    assert c.is_winner('x')

    last_move = c.remove()
    c.draw()
    assert not c.is_winner('x')

    c.draw()
    c.place('x', last_move)
    assert c.is_winner('x')


def test_player_functions():
    c = ConnectFour()
    c.place('o', 1)
    c.place('o', 1)

    c.place('x', 2)
    c.place('o', 2)
    c.place('o', 2)
    c.place('x', 2)

    c.place('o', 3)
    c.place('o', 3)
    c.place('o', 3)
    c.place('x', 3)

    c.place('o', 4)
    c.place('x', 4)
    c.place('o', 4)
    c.place('x', 4)
    c.place('x', 4)

    c.place('x', 5)
    c.place('x', 5)
    c.place('x', 5)
    c.place('o', 5)
    c.place('x', 5)
    c.place('x', 5)

    c.draw()

    assert c.possible_win('o', n=3) == 4
    assert c.possible_win('x', n=3) == 1


def test_case_2():
    c = ConnectFour()
    c.place('x', 1)
    c.place('o', 1)

    c.place('o', 2)
    c.place('o', 2)

    c.place('o', 3)

    c.place('x', 4)
    c.place('o', 4)

    c.place('x', 5)
    c.place('x', 5)

    c.place('x', 6)

    c.draw()

    p = Player('x', 'o', c)
    assert c.possible_win('o', n=3) == 1
    assert p.turn() == 3


def build_game(board):
    c = ConnectFour()
    for row in reversed(board):
        for i, mark in enumerate(row):
            if mark is not None:
                c.place(mark, i)

    return c


def test_row():
    c = build_game([
        ['o', 'o', 'o', 'x', 'x', 'x', 'x'],
        ['o', 'o', 'x', 'x', 'x', 'x', 'x'],
        ['o', 'x', 'o', 'x', 'o', 'o', 'o'],
        ['o', 'o', 'x', 'o', 'o', 'x', 'o'],
        ['o', 'x', 'x', 'o', 'o', 'x', 'x'],
        ['x', 'o', 'o', 'x', 'x', 'x', 'x'],
    ])
    assert c.row(0) == ['o', 'o', 'o', 'x', 'x', 'x', 'x']
    assert c.row(1) == ['o', 'o', 'x', 'x', 'x', 'x', 'x']
    assert c.row(2) == ['o', 'x', 'o', 'x', 'o', 'o', 'o']
    assert c.row(3) == ['o', 'o', 'x', 'o', 'o', 'x', 'o']
    assert c.row(4) == ['o', 'x', 'x', 'o', 'o', 'x', 'x']
    assert c.row(5) == ['x', 'o', 'o', 'x', 'x', 'x', 'x']


def test_column():
    c = build_game([
        ['o', 'o', 'o', 'x', 'x', 'x', 'x'],
        ['o', 'o', 'x', 'x', 'x', 'x', 'x'],
        ['o', 'x', 'o', 'x', 'o', 'o', 'o'],
        ['o', 'o', 'x', 'o', 'o', 'x', 'o'],
        ['o', 'x', 'x', 'o', 'o', 'x', 'x'],
        ['x', 'o', 'o', 'x', 'x', 'x', 'x'],
    ])
    assert c.column(0) == ['o', 'o', 'o', 'o', 'o', 'x']
    assert c.column(1) == ['o', 'o', 'x', 'o', 'x', 'o']
    assert c.column(2) == ['o', 'x', 'o', 'x', 'x', 'o']
    assert c.column(3) == ['x', 'x', 'x', 'o', 'o', 'x']
    assert c.column(4) == ['x', 'x', 'o', 'o', 'o', 'x']
    assert c.column(5) == ['x', 'x', 'o', 'x', 'x', 'x']
    assert c.column(6) == ['x', 'x', 'o', 'o', 'x', 'x']


def test_diagonal():
    c = build_game([
        ['o', 'o', 'o', 'x', 'x', 'x', 'x'],
        ['o', 'o', 'x', 'x', 'x', 'x', 'x'],
        ['o', 'x', 'o', 'x', 'o', 'o', 'o'],
        ['o', 'o', 'x', 'o', 'o', 'x', 'o'],
        ['o', 'x', 'x', 'o', 'o', 'x', 'x'],
        ['x', 'o', 'o', 'x', 'x', 'x', 'x'],
    ])
    assert c.diagonal(0) == ['o', 'o', 'o', 'o', 'o', 'x']
    assert c.diagonal(1) == ['o', 'x', 'x', 'o', 'x', 'x']
    assert c.diagonal(2) == ['o', 'x', 'o', 'x', 'x']
    assert c.diagonal(3) == ['x', 'x', 'o', 'o']
    assert c.diagonal(4) == ['x', 'x', 'o']
    assert c.diagonal(5) == ['x', 'x']
    assert c.diagonal(6) == ['x']
    assert c.diagonal(7) == ['o', 'x', 'x', 'o', 'x']
    assert c.diagonal(8) == ['o', 'o', 'x', 'x']
    assert c.diagonal(9) == ['o', 'x', 'o']
    assert c.diagonal(10) == ['o', 'o']
    assert c.diagonal(11) == ['x']


def test_anti_diagonal():
    c = build_game([
        ['o', 'o', 'o', 'x', 'x', 'x', 'x'],
        ['o', 'o', 'x', 'x', 'x', 'x', 'x'],
        ['o', 'x', 'o', 'x', 'o', 'o', 'o'],
        ['o', 'o', 'x', 'o', 'o', 'x', 'o'],
        ['o', 'x', 'x', 'o', 'o', 'x', 'x'],
        ['x', 'o', 'o', 'x', 'x', 'x', 'x'],
    ])
    assert c.anti_diagonal(0) == ['o']
    assert c.anti_diagonal(1) == ['o', 'o']
    assert c.anti_diagonal(2) == ['o', 'o', 'o']
    assert c.anti_diagonal(3) == ['x', 'x', 'x', 'o']
    assert c.anti_diagonal(4) == ['x', 'x', 'o', 'o', 'o']
    assert c.anti_diagonal(5) == ['x', 'x', 'x', 'x', 'x', 'x']
    assert c.anti_diagonal(6) == ['x', 'x', 'o', 'o', 'x', 'o']
    assert c.anti_diagonal(7) == ['x', 'o', 'o', 'o', 'o']
    assert c.anti_diagonal(8) == ['o', 'x', 'o', 'x']
    assert c.anti_diagonal(9) == ['o', 'x', 'x']
    assert c.anti_diagonal(10) == ['x', 'x']
    assert c.anti_diagonal(11) == ['x']


def test_player_row_1():
    """
    x x x _ o o o
    """
    c = ConnectFour()
    c.place('x', 0)
    c.place('x', 1)
    c.place('x', 2)
    c.place('o', 4)
    c.place('o', 5)
    assert c.possible_win('x', n=3) == 1
    assert c.possible_win('o', n=3) == 0


def test_player_row_2():
    """
    x x _ x o o o
    """
    c = ConnectFour()
    c.place('x', 0)
    c.place('x', 1)
    c.place('x', 3)
    c.place('o', 4)
    c.place('o', 5)

    assert c.possible_win('x', n=3) == 1
    assert c.possible_win('o', n=3) == 0


def test_player_row_3():
    """
    x _ x x o o o
    """
    c = ConnectFour()
    c.place('x', 0)
    c.place('x', 2)
    c.place('x', 3)
    c.place('o', 4)
    c.place('o', 5)

    assert c.possible_win('x', n=3) == 1
    assert c.possible_win('o', n=3) == 0


def test_player_row_4():
    """
    _ x x x o o o
    """
    c = ConnectFour()
    c.place('x', 1)
    c.place('x', 2)
    c.place('x', 3)
    c.place('o', 4)
    c.place('o', 5)

    assert c.possible_win('x', n=3) == 1
    assert c.possible_win('o', n=3) == 0


def test_player_col():
    """
    x
    x _ o
    x _ o

    """
    c = ConnectFour()
    c.place('x', 0)
    c.place('x', 0)
    c.place('x', 0)
    c.place('o', 2)
    c.place('o', 2)

    assert c.possible_win('x', n=3) == 1
    assert c.possible_win('o', n=3) == 0


def test_player_diagonal1():
    c = build_game([
        [None, 'o', 'o', 'o'],
        ['o', 'x', 'o', 'o'],
        ['o', 'o', 'x', 'o'],
        ['o', 'o', 'o', 'x'],
    ])

    assert c.possible_win('x', n=3) == 1


def test_player_diagonal2():
    c = build_game([
        ['x', None, 'o', 'o'],
        ['o', None, 'o', 'o'],
        ['o', 'o', 'x', 'o'],
        ['o', 'o', 'o', 'x'],
    ])

    assert c.possible_win('x', n=3) == 1


def test_player_diagonal3():
    c = build_game([
        ['x', 'o', None, 'o'],
        ['o', 'x', None, 'o'],
        ['o', 'o', None, 'o'],
        ['o', 'o', 'o', 'x'],
    ])

    assert c.possible_win('x', n=3) == 1


def test_player_diagonal4():
    c = build_game([
        ['x', 'o', 'o', None],
        ['o', 'x', 'o', None],
        ['o', 'o', 'x', None],
        ['o', 'o', 'o', None],
    ])
    assert c.possible_win('x', n=3) == 1


def test_player_anti_diagonal1():
    c = build_game([
        ['o', 'o', 'o'],
        ['o', 'o', 'x', 'o'],
        ['o', 'x', 'o', 'o'],
        ['x', 'o', 'o', 'o'],
    ])

    assert c.possible_win('x', n=3) == 1


def test_player_anti_diagonal2():
    c = build_game([
        ['o', 'o', None, 'x'],
        ['o', 'o', None, 'o'],
        ['o', 'x', 'o', 'o'],
        ['x', 'o', 'o', 'o'],
    ])

    assert c.possible_win('x', n=3) == 1


def test_player_anti_diagonal3():
    c = build_game([
        ['o', None, 'o', 'x'],
        ['o', None, 'x', 'o'],
        ['o', None, 'o', 'o'],
        ['x', 'o', 'o', 'o'],
    ])

    assert c.possible_win('x', n=3) == 1


def test_player_anti_diagonal4():
    c = build_game([
        [None, 'o', 'o', 'x'],
        [None, 'o', 'x', 'o'],
        [None, 'x', 'o', 'o'],
        [None, 'o', 'o', 'o'],
    ])
    assert c.possible_win('x', n=3) == 2


def test_board_full():
    c = build_game([
        [None, 'o', 'x', 'o', 'o', 'x', 'x'],
        [None, 'x', 'o', 'x', 'x', 'x', 'o'],
        ['o', 'x', 'x', 'o', 'o', 'x', 'o'],
        ['x', 'o', 'o', 'x', 'x', 'o', 'x'],
        ['x', 'x', 'o', 'o', 'o', 'x', 'x'],
        ['o', 'o', 'o', 'x', 'o', 'o', 'x'],
    ])
    p = Player('x', 'o', c)
    assert p.turn() == 0

    c.place('x', 0)

    assert p.turn() == 0


def test_board_full2():
    c = build_game([
        ['o', None, 'o', None, 'x', 'o', 'o'],
        ['x', None, 'o', None, 'x', 'o', 'x'],
        ['o', 'x', 'x', 'o', 'o', 'o', 'x'],
        ['x', 'o', 'o', 'x', 'x', 'x', 'o'],
        ['o', 'x', 'x', 'o', 'o', 'x', 'x'],
        ['x', 'x', 'o', 'x', 'x', 'o', 'o'],
    ])
    p = Player('x', 'o', c)
    spot = p.turn()
    assert (spot == 1 or spot == 3)

    c.place('x', 1)
    c.place('x', 1)

    assert p.turn() == 3


if __name__ == '__main__':
    test_board_full2()
