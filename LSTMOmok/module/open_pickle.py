import pickle
import numpy as np

def OpenPickle(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
    boards = OpenPickle('boards.pkl')
    print(len(boards))
    for i in range(100):
        print(np.array(boards[i]).shape)
