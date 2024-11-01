import numpy as np
import os
import glob
import pickle

def LoadSequence(path: str) -> list[str]:
    with open(path, 'r') as f:
        sequence = f.read().splitlines()

    sequence = sequence[1:-3]

    return sequence

def LoadBoard(paths: list[str]) -> list[list[list[list[int]]]]:
    boards: list[list[list[list[int]]]] = []
    for path in paths:
        sequence = LoadSequence(path)

        board = [[0 for _ in range(15)] for _ in range(15)]
        turn = 1
        first = []
        for line in sequence:
            l = line.split(',')
            if len(l) > 2:
                x, y= int(l[0]), int(l[1])
            else:
                continue

            if x == np.nan or y == np.nan:
                continue

            if x > 15 or y > 15:
                break

            board[x-1][y-1] = turn
            turn = -turn

            if np.array(board).shape != (15, 15):
                continue

            first.append([row[:] for row in board])

        second = []
        third = []
        fourth = []
        for f in first:
            array = np.array(f)
            second.append(np.rot90(array, k=1).tolist())
            third.append(np.rot90(array, k=2).tolist())
            fourth.append(np.rot90(array, k=3).tolist())

        boards.append(first)
        boards.append(second)
        boards.append(third)
        boards.append(fourth)

    with open('boards3.pkl', 'wb') as f:
        pickle.dump(boards, f)

if __name__ == "__main__":
    paths = glob.glob('Standard/*.psq')
    print(len(paths))
    LoadBoard(paths)
