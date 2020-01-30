# Connect four

## How to play
```
python main.py --player1 <player> --player2 <player> --difficulty <difficulty>
```

player can be one of the following:
* `computer` - smart player
* `random` - random player
* `human` - human player.
* `monte_carlo` - monte_carlo player

difficulty can be from 1 to 10. notice that above 6, the computer takes time to play each turn.

For exmaple:
```
python main.py --player1 human --player2 computer
```

the defaults are `player1=computer`, `player2=human` and `difficulty=5`