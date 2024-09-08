from keras import models, layers
import tensorflow as tf
import numpy as np
import json
import os


# Gene Algorithms for Additional Learning
def create_model1():
    with open(os.path.join("model", "default2.json"), "r") as f:
        model_layer = f.read()
    
    model = models.model_from_json(model_layer)

    return model

def inital_population(size):
    population = []
    base_model1 = create_model1()
    base_model1.load_weights(os.path.join("model", "default2.h5"))
    base_weight1 = base_model1.get_weights()

    for _ in range(int(size)):
        new_model = create_model1()
        new_weights = []

        weight_index = 0  # 가중치 텐서의 인덱스를 추적

        for layer in base_model1.layers:
            num_weights = len(layer.get_weights())
            if num_weights == 0:  # 가중치가 없는 레이어는 건너뜁니다.
                continue
            for i in range(num_weights):
                w = base_weight1[weight_index]
                if 'batch_normalization' not in layer.name:
                    pert = np.random.randn(*w.shape) * 0.01
                    new_w = w + pert
                else:
                    new_w = w  # batch normalization의 경우 변경하지 않음
                new_weights.append(new_w)
                weight_index += 1

        new_model.set_weights(new_weights)
        new_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
        population.append(new_model)

    return population

def get_non_bn_layer_names(model):
    return [layer for layer in model.layers]

def mutate(weights, layer_names, rate=0.05):
    new_weights = []
    weight_index = 0
    
    for layer in layer_names:
        num_weights = len(layer.get_weights())
        if num_weights == 0:
            continue
        
        for i in range(num_weights):
                w = weights[weight_index]
                if 'batch_normalization' not in layer.name:
                    pert = 0
                    if np.random.rand() < rate:
                        pert = np.random.randn(*w.shape) * 0.01
                    new_w = w + pert
                else:
                    new_w = w  # batch normalization의 경우 변경하지 않음
                new_weights.append(new_w)
                weight_index += 1


    # 디버깅을 위한 로그 추가
    print(f"[Mutate] Generated {len(new_weights)} weights")

    return new_weights

def crossover(parent1, parent2, layer_names):
    parent1_weights = parent1.get_weights()
    parent2_weights = parent2.get_weights()
    child_weights = []
    weight_index = 0

    for layer in layer_names:
        num_weights = len(layer.get_weights())
        if num_weights == 0:
            continue
        
        for i in range(num_weights):
            shape = parent1_weights[weight_index].shape
            mask = np.random.rand(*shape) > 0.5
            child_w = np.where(mask, parent1_weights[weight_index], parent2_weights[weight_index])
            child_weights.append(child_w)
            weight_index += 1
            

    # 디버깅을 위한 로그 추가
    print(f"[Crossover] Generated {len(child_weights)} weights")

    return child_weights

from glob import glob 
from tqdm import tqdm

def evaluate_fitness(model, game_state):
    fit = 0

    for board in game_state:
        board = np.array(board).reshape((15, 15)).tolist()
        input_board = np.expand_dims(board, axis=(0, -1)).astype(np.float32)
        output = model.predict(input_board, verbose=0).squeeze()

        if np.isnan(output).any():
            print("NaN detected in model output")
            return float('-inf')  # NaN이 감지되면 매우 낮은 피트니스 값을 반환

        output = output.reshape((15, 15))
        tmp_board = np.array(board).reshape((15, 15))
        mod_output = output.copy()

        for i in range(15):
            for j in range(15):
                if tmp_board[i][j] != 0:
                    mod_output[i][j] -= 1e5

        y, x = np.unravel_index(np.argmax(mod_output), mod_output.shape)
        tx, ty = x, y

        if board[ty][tx] != 0:
            print("헉")
            continue  # 유효하지 않은 위치인 경우 이 보드는 무시

        copy_board = [arr[:] for arr in board]
        fit += calculate_fitness(tx, ty, copy_board)

    return fit

def calculate_fitness(_x, _y, board: list):
    def isValid(ux, uy):
        if ux >= 0 and ux < 15 and uy >= 0 and uy < 15:
            return True
        return False

    isB, isW = 0, 0
    for WhyisntGeneWorking in board:
        isB += np.count_nonzero(WhyisntGeneWorking == 1)
        isW += np.count_nonzero(WhyisntGeneWorking == -1)


    turn = 1
    notTurn = -1
    if isB == isW:
        turn = -1
        notTurn = 1

    def countBoth(dx, dy, turn, notTurn):
        x, y = _x, _y
        cnt1, cnt2 = 0, 0

        # 막혔음 - True
        isBreak = False
        blankCnt = 0

        cnt = 1 if board[y][x] == turn else 0

        for i in range(1, 15):
            x += dx
            y += dy

            if isValid(x, y):
                if board[y][x] == turn:
                    cnt1 += 1
                    if (isValid(x + dx, y + dy) and board[y][x] == notTurn) or not isValid(x + dx, y + dy):
                        cnt1 = 0
                        isBreak = True
                        break
                elif board[y][x] == 0:
                    blankCnt += 1
                    if isValid(x - dx, y - dy) and board[y - dy][x - dx] == 0:
                        break
                    if blankCnt == 2:
                        break
                elif isValid(_x + dx, _y + dy) and board[_y + dy][_x + dx] == notTurn:
                    isBreak = True
                    break
                elif board[y][x] == notTurn and cnt1 == 0:
                    break

        x, y = _x, _y
        blankCnt = 0

        for i in range(1, 15):
            x -= dx
            y -= dy

            if isValid(x, y):
                if board[y][x] == turn:
                    cnt2 += 1
                    if (isValid(x - dx, y - dy) and board[y][x] == notTurn) or not isValid(x - dx, y - dy):
                        cnt2 = 0
                        isBreak = True
                        break
                elif board[y][x] == 0:
                    blankCnt += 1
                    if isValid(x + dx, y + dy) and board[y + dy][x + dx] == 0:
                        break
                    if blankCnt == 2:
                        break
                elif isValid(_x - dx, _y - dy) and board[_y - dy][_x - dx] == notTurn:
                    isBreak = True
                    break
                elif board[y][x] == notTurn and cnt2 == 0:
                    break

        return {"b": isBreak, "cnt1": cnt1, "cnt2": cnt2, "check": cnt}
        
    
    def isDefend(dx, dy):
        dic = countBoth(dx, dy, notTurn, turn)
        score = dic["cnt1"] + dic["cnt2"]

        return (score + 2) / 10

        



    fitness_score = 0

    move = [(1, -1), (1, 0), (1, 1), (0, 1)]
    turn_count = []
    defend_count = []
    for dx, dy in move:
        dic = countBoth(dx, dy, turn, notTurn)
        val = dic["cnt1"] + dic["cnt2"] + dic["check"] 

        if dic["b"]:
            val -= 2
        
        turn_count.append(val / 10)
        defend_count.append(isDefend(dx, dy))

    for a, b in zip(turn_count, defend_count):
        fitness_score += (a + b)
    
    fitness_score += (float(max(turn_count)) * 1.3 + float(max(defend_count)) * 1.3)


    return fitness_score

def select_data(path):
    data = glob(os.path.join(path, '*.npz'))
    all_boards = [np.load(file)['boards'] for file in data]

    all_boards_combined = np.vstack(all_boards)
    non_zero_counts = np.count_nonzero(all_boards_combined, axis=(1, 2))

    boards_10 = [board for board, count in tqdm(zip(all_boards_combined, non_zero_counts)) if count < 10]
    boards_30 = [board for board, count in tqdm(zip(all_boards_combined, non_zero_counts)) if 10 <= count < 30]
    boards_50 = [board for board, count in tqdm(zip(all_boards_combined, non_zero_counts)) if 30 <= count < 50]
    boards_over_50 = [board for board, count in tqdm(zip(all_boards_combined, non_zero_counts)) if 50 <= count <= 200]

    selected_boards_10 = np.random.choice(len(boards_10), 7, replace=False)
    selected_boards_30 = np.random.choice(len(boards_30), 13, replace=False)
    selected_boards_50 = np.random.choice(len(boards_50), 10, replace=False)
    selected_boards_over_50 = np.random.choice(len(boards_over_50), 5, replace=False)

    # 선택된 보드를 game_state에 추가
    game_state = [boards_10[i] for i in selected_boards_10]
    game_state += [boards_30[i] for i in selected_boards_30]
    game_state += [boards_50[i] for i in selected_boards_50]
    game_state += [boards_over_50[i] for i in selected_boards_over_50]

    return game_state


def main_genetic(size, path):
    pop = 50
    population = inital_population(pop)
    layer_names = get_non_bn_layer_names(population[0])

    game_state = select_data(path)
    print("successfully selected the data")

    for generation in tqdm(range(size)):
        fitness_score = []

        for model in population:
            fitness = evaluate_fitness(model, game_state)
            fitness_score.append((model, fitness))

        fitness_score.sort(key=lambda x: x[1], reverse=True)
        top_individuals = fitness_score[:pop // 2]

        new_population = []
        for i in range(0, len(top_individuals) - 1, 2):
            parent1, _ = top_individuals[i]
            parent2, _ = top_individuals[i + 1]
            
            child_weights = crossover(parent1, parent2, layer_names)
            child_model = create_model1()
            new_weights = mutate(child_weights, layer_names)

            # Check if the weight count matches before setting weights
            expected_length = len(child_model.get_weights())
            if len(new_weights) != expected_length:
                print(f"[Error] Length mismatch in generation {generation + 1}: Expected {expected_length}, got {len(new_weights)}")
                print(f"Layer Names: {layer_names}")
                print(f"Parent 1 Weights: {len(parent1.get_weights())}, Parent 2 Weights: {len(parent2.get_weights())}")
                raise ValueError(f"Generated weights length ({len(new_weights)}) does not match model's expected length ({expected_length})")

            child_model.set_weights(new_weights)
            child_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
            
            new_population.append(child_model)

        population = [model for model, _ in top_individuals] + new_population

        if len(population) < pop:
            for _ in range(pop - len(population)):
                fir = np.random.randint(0, len(top_individuals))
                sec = np.random.randint(0, len(top_individuals))

                parent1, _ = top_individuals[fir]
                parent2, _ = top_individuals[sec]

                child_weights = crossover(parent1, parent2, layer_names)
                child_model = create_model1()
                new_weights = mutate(child_weights, layer_names)

                # Check if the weight count matches before setting weights
                expected_length = len(child_model.get_weights())
                if len(new_weights) != expected_length:
                    print(f"[Error] Length mismatch in generation {generation + 1}: Expected {expected_length}, got {len(new_weights)}")
                    raise ValueError(f"Generated weights length ({len(new_weights)}) does not match model's expected length ({expected_length})")

                child_model.set_weights(new_weights)
                child_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
                
                population.append(child_model)

        print(f"Generation {generation + 1} Population size: {len(population)}")

    best_model = max(fitness_score, key=lambda x: x[1])[0]
    print(f"Best fitness score: {max(fitness_score, key=lambda x: x[1])[1]}")
    return best_model
