"""Connect four game infrastructure implementation."""
import sys
import itertools
from io import StringIO
from collections import defaultdict
from contextlib import contextmanager


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

def color(func):
    """Color each player's character in different color."""

    def wrapper(self):
        with mock_stdout() as output:
            func(self)
            char_to_color = {}
            colors = ['fail','green']
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