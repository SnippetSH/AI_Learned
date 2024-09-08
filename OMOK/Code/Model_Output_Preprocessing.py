import numpy as np
import json


# Control Output to Ensure the Performance of AI 
def remove_nonzero(output, board):
    for i in range(15):
        for j in range(15):
            if board[i][j] != 0:
                output[i][j] -= 1e5

    return output

def predict(model_1, model_2, model_3, model_4, board, session_id, session_predictions):
    board = json.loads(board)

    target_x, target_y = next(((j, i) for i in range(15) for j in range(15) if board[i][j] == 1), (0, 0))
    tmp_cnt = sum(board[i][j] == 1 for i in range(15) for j in range(15))
    white_cnt = sum(board[i][j] == -1 for i in range(15) for j in range(15))

    isFirst = (tmp_cnt == 1 and white_cnt == 0)
    if isFirst:
        move = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
        r = np.random.randint(0, 8)
        if target_y == 7 and target_x == 7:
            return target_x + move[r][0], target_y + move[r][1]
        else:
            return 7, 7
        
    if tmp_cnt == 0 and white_cnt == 0:
        return 7, 7

    input_board = np.expand_dims(board, axis=(0, -1)).astype(np.float32)
    output = model_1.predict(input_board, verbose=0).squeeze()
    output = output.reshape((15, 15))
    mod_output = np.array([arr[:] for arr in output])
    nonzero = np.count_nonzero(board)

    input_board = np.expand_dims(board, axis=(0, -1)).astype(np.float32)
    output1 = model_2.predict(input_board, verbose=0).squeeze()
    output1 = output1.reshape((15, 15))
    mod_output1 = np.array([arr[:] for arr in output1])

    input_board = np.expand_dims(board, axis=(0, -1)).astype(np.float32)
    output2 = model_3.predict(input_board, verbose=0).squeeze()
    output2 = output2.reshape((15, 15))
    mod_output2 = np.array([arr[:] for arr in output2])

    input_board = np.expand_dims(board, axis=(0, -1)).astype(np.float32)
    output3 = model_4.predict(input_board, verbose=0).squeeze()
    output3 = output3.reshape((15, 15))
    mod_output3 = np.array([arr[:] for arr in output3])
    
    if nonzero < 5:
        mod_output[[0, 1, 2, 3, 4, 10, 11, 12, 13, 14], :] -= 1e-7
        mod_output[:, [0, 1, 2, 3, 4, 10, 11, 12, 13, 14]] -= 1e-7

        mod_output1[[0, 1, 2, 3, 4, 10, 11, 12, 13, 14], :] -= 1e-7
        mod_output1[:, [0, 1, 2, 3, 4, 10, 11, 12, 13, 14]] -= 1e-7

        mod_output2[[0, 1, 2, 3, 4, 10, 11, 12, 13, 14], :] -= 1e-7
        mod_output2[:, [0, 1, 2, 3, 4, 10, 11, 12, 13, 14]] -= 1e-7

        mod_output3[[0, 1, 2, 3, 4, 10, 11, 12, 13, 14], :] -= 1e-7
        mod_output3[:, [0, 1, 2, 3, 4, 10, 11, 12, 13, 14]] -= 1e-7

    elif nonzero < 10:
        mod_output[[0, 1, 2, 10, 11, 12], :] -= 1e-7
        mod_output[:, [0, 1, 2, 10, 11, 12]] -= 1e-7

        mod_output1[[0, 1, 2, 10, 11, 12], :] -= 1e-7
        mod_output1[:, [0, 1, 2, 10, 11, 12]] -= 1e-7

        mod_output2[[0, 1, 2, 10, 11, 12], :] -= 1e-7
        mod_output2[:, [0, 1, 2, 10, 11, 12]] -= 1e-7

        mod_output3[[0, 1, 2, 10, 11, 12], :] -= 1e-7
        mod_output3[:, [0, 1, 2, 10, 11, 12]] -= 1e-7
        

    if session_id in session_predictions and len(session_predictions[session_id]) > 0:
        while (x, y) in session_predictions[session_id]:
            mod_output[y][x] -= 1e2
            mod_output1[y][x] -= 1e2
            mod_output2[y][x] -= 1e2
            mod_output3[y][x] -= 1e2

    # test
    # print(np.max(mod_output), np.sort(mod_output.flatten())[-2])
    isDanger, dangerInfo = check_four(board)
    BBBB = (-1, -1)
    X1, Y1, X2, Y2 = -1, -1, -1, -1
    DX, DY = 0, 0
    three = False
    if isDanger:
        positions = dangerInfo["positions"]
        print("type", dangerInfo["type"])
        print("pos", positions)
        x1, y1 = positions[0]
        x2, y2 = positions[-1]
        X1, Y1, X2, Y2 = x1, y1, x2, y2
        dx, dy = dangerInfo["dx"], dangerInfo["dy"]
        DX, DY = dx, dy
        
        if dangerInfo["type"] == "four":
            
            blank = (-1, -1)
            for i in range(len(positions) - 1):
                if max(positions[i + 1][0] - positions[i][0], positions[i + 1][1] - positions[i][1]) > 1:
                    blank = positions[i]
                    break

            print("여기 빔", blank, dx, dy)
            if blank != (-1, -1):
                if board[blank[1] + dy][blank[0] + dx] == 0:
                    return blank[0] + dx, blank[1] + dy
                
            elif board[y1 - dy][x1 - dx] == 0:
                return x1 - dx, y1 - dy
            elif board[y2 + dy][x2 + dx] == 0:
                return x2 + dx, y2 + dy
            
        elif dangerInfo["type"] == "three":
            if board[y1 - dy][x1 - dx] == 0 and board[y2 + dy][x2 + dx] == 0:
                three = True
                print("여기 3임", positions[0], positions[-1], dx, dy)
                blank = (-1, -1)
                for i in range(len(positions) - 1):
                    if max(positions[i + 1][0] - positions[i][0], positions[i + 1][1] - positions[i][1]) > 1:
                        blank = positions[i]
                        BBBB = blank
                        break

                if blank != (-1, -1):
                    if board[blank[1] + dy][blank[0] + dx] == 0:
                        tmp1 = np.max(mod_output2)
                        tmpXY1 = np.unravel_index(np.argmax(mod_output2), mod_output2.shape)
                        
                        mod_output2[tmpXY1[0]][tmpXY1[1]] = -1e5
                        mod_output2[blank[1] + dy][blank[0] + dx] = tmp1
                        
                        if np.random.random() > 0.5:
                            tmp2 = np.max(mod_output3)
                            tmpXY2 = np.unravel_index(np.argmax(mod_output3), mod_output3.shape)
                            mod_output3[y2 + dy][x2 + dx] = tmp2
                            mod_output3[tmpXY2[0]][tmpXY2[1]] = -1e5
                        else:
                            tmp2 = np.max(mod_output3)
                            tmpXY2 = np.unravel_index(np.argmax(mod_output3), mod_output3.shape)
                            mod_output3[y1 - dy][x1 - dx] = tmp2
                            mod_output3[tmpXY2[0]][tmpXY2[1]] = -1e5
                else:
                    if board[y1 - dy][x1 - dx] == 0:
                        tmp1 = np.max(mod_output2)
                        tmpXY1 = np.unravel_index(np.argmax(mod_output2), mod_output2.shape)

                        mod_output2[tmpXY1[0]][tmpXY1[1]] = -1e5
                        mod_output2[y1 - dy][x1 - dx] = tmp1
                    if board[y2 + dy][x2 + dx] == 0:
                        tmp2 = np.max(mod_output3)
                        tmpXY2 = np.unravel_index(np.argmax(mod_output3), mod_output3.shape)

                        mod_output3[tmpXY2[0]][tmpXY2[1]] = -1e5
                        mod_output3[y2 + dy][x2 + dx] = tmp2

    mod_output = remove_nonzero(mod_output, board)
    mod_output1 = remove_nonzero(mod_output1, board)
    mod_output2 = remove_nonzero(mod_output2, board)
    mod_output3 = remove_nonzero(mod_output3, board)

    predicts = set()
    predicts.add(np.unravel_index(np.argmax(mod_output), mod_output.shape))
    predicts.add(np.unravel_index(np.argmax(mod_output1), mod_output1.shape))
    predicts.add(np.unravel_index(np.argmax(mod_output2), mod_output2.shape))
    predicts.add(np.unravel_index(np.argmax(mod_output3), mod_output3.shape))

    if len(list(predicts)) > 1:
        y, x = predict_tree(model_1, model_2, model_3, model_4, board, list(predicts))

        if three:
            directions = [(1, 0), (0, 1), (1, 1), (1, -1), (-1, 0), (0, -1), (-1, -1), (-1, 1)]
            black_cnt = sum(board[i][j] == 1 for i in range(15) for j in range(15))
            white_cnt = sum(board[i][j] == -1 for i in range(15) for j in range(15))
            turn = 1 if black_cnt == white_cnt else -1
            
            for dx, dy in directions:
                count = 1
                _x, _y = x, y
                
                b = 0
                while True:
                    _x += dx
                    _y += dy
                    if isValid(_x, _y):
                        if board[_y][_x] == turn:
                            count += 1
                        elif board[_y][_x] == 0 and isValid(_x + dx, _y + dy) and board[_y + dy][_x + dx] == turn and b == 0:
                            b += 1
                        elif board[_y][_x] == 0 and b >= 1:
                            break
                        elif board[_y][_x] == -turn:
                            break
                        else:
                            break
                    else:
                        break                    
                
                if count != 4:
                    if BBBB != (-1, -1):
                        if board[BBBB[1] + DY][BBBB[0] + DX] == 0:
                            x, y = BBBB[0] + DX, BBBB[1] + DY
                    else:
                        if np.random.random() > 0.5:
                            x, y = X1 - DX, Y1 - DY
                        else:
                            x, y = X2 + DX, Y2 + DY

    else:
        y, x = list(predicts)[0]

    if session_id not in session_predictions:
        session_predictions[session_id] = []
    session_predictions[session_id].append((x, y))
    return x, y

def check_four(board):
    directions = [(1, 0), (0, 1), (1, 1), (1, -1), (-1, 0), (0, -1), (-1, -1), (-1, 1)]
    black_cnt = sum(board[i][j] == 1 for i in range(15) for j in range(15))
    white_cnt = sum(board[i][j] == -1 for i in range(15) for j in range(15))

    turn = -1 if black_cnt == white_cnt else 1

    dangers = []
    for i in range(15):
        for j in range(15):
            if board[i][j] != turn:
                continue
            for dx, dy in directions:
                count = 1
                x, y = j, i
                blank = 0
                isBreak = False
                positions = [(j, i)]
                while True:
                    x += dx
                    y += dy
                    if isValid(x, y):
                        if board[y][x] == turn:
                            count += 1
                            positions.append((x, y))
                        elif board[y][x] == 0 and isValid(x + dx, y + dy) and board[y + dy][x + dx] == turn and blank == 0:
                            blank += 1
                        elif board[y][x] == 0 and blank == 1:
                            break
                        elif board[y][x] == -turn:
                            _x, _y = x, y
                            while True:
                                _x -= dx
                                _y -= dy
                                if isValid(_x, _y) and board[_y][_x] == -turn:
                                    isBreak = True
                                    break
                                elif isValid(_x, _y) and board[_y][_x] == 0 and isValid(_x - dx, _y - dy) and board[_y - dy][_x - dx] == 0:
                                    break
                                elif not isValid(_x, _y):
                                    break
                            break
                        else:
                            break
                    else:
                        break
                if not isBreak and count == 4:
                    # print("너 4임", x - dx, y - dy)
                    dangers.append((True, {"type": "four", "positions": positions, "dx": dx, "dy": dy}))
                elif not isBreak and count == 3:
                    # print("너 3임", x - dx, y - dy)
                    dangers.append((True, {"type": "three", "positions": positions, "dx": dx, "dy": dy}))

    if len(dangers) > 0:
        for danger in dangers:
            if danger[1]["type"] == "four":
                return danger
            
        new_dangers = []
        for danger in dangers:
            x1, y1 = danger[1]["positions"][0]
            x2, y2 = danger[1]["positions"][-1]
            dx, dy = danger[1]["dx"], danger[1]["dy"]
            
            print(board[x1 - dx][y1 - dy], board[x2 + dx][y2 + dy])
            if board[x1 - dx][y1 - dy] == 0 and board[x2 + dx][y2 + dy] == 0:
                new_dangers.append(danger)

        dangers = new_dangers

        print(dangers)
        if len(dangers) > 0:
            return dangers[0]

    return False, None
                
def isValid(x, y):
    if 0 <= x < 15 and 0 <= y < 15:
        return True
    return False

def isWin(x, y, board, turn):
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
    for dx, dy in directions:
        count = 1  # 현재 위치 포함
        for direction in [1, -1]:
            nx, ny = x, y
            while True:
                nx += direction * dx
                ny += direction * dy
                if isValid(nx, ny) and board[ny][nx] == turn:
                    count += 1
                else:
                    break
            if count >= 5:
                # print(count)
                return True
    return False
    

# Maybe Need
# This function implements simulation with other Omok models 
from tqdm import tqdm
def predict_tree(model_1, model_2, model_3, model_4, board, predicts):
    black_cnt = sum(board[i][j] == 1 for i in range(15) for j in range(15))
    white_cnt = sum(board[i][j] == -1 for i in range(15) for j in range(15))

    turn = 1 if black_cnt == white_cnt else -1

    win_scores = [0 for _ in range(len(predicts))]
    lose_scores = [0 for _ in range(len(predicts))]
    length = []
    # 현재 턴에 둘 수 있는 10개의 위치 (predicts)
    for i, predict in enumerate(tqdm(predicts)):
        cur_board = np.array(board).reshape((15, 15))
        x, y = predict
        cur_board[y][x] = turn
        input_board = np.expand_dims(cur_board, axis=(0, -1)).astype(np.float32)
        output = model_1.predict(input_board, verbose=0).squeeze()
        output = output.reshape((15, 15))
        output = remove_nonzero(output, cur_board)

        output1 = model_2.predict(input_board, verbose=0).squeeze()
        output1 = output1.reshape((15, 15))
        output1 = remove_nonzero(output1, cur_board)

        output2 = model_3.predict(input_board, verbose=0).squeeze()
        output2 = output2.reshape((15, 15))
        output2 = remove_nonzero(output2, cur_board)

        output3 = model_4.predict(input_board, verbose=0).squeeze()
        output3 = output3.reshape((15, 15))
        output3 = remove_nonzero(output3, cur_board)

        # 상대 턴 (시뮬레이션을 돌리기 위한 상대가 둘 수 있는 10개의 위치)
        new_predicts = set()
        new_predicts.add(np.unravel_index(np.argmax(output), output.shape))
        new_predicts.add(np.unravel_index(np.argmax(output1), output1.shape))
        new_predicts.add(np.unravel_index(np.argmax(output2), output2.shape))
        new_predicts.add(np.unravel_index(np.argmax(output3), output3.shape))

        new_predicts = list(new_predicts)
        length.append(len(new_predicts))
        # print(len(new_predicts))
        del output, output1, output2, output3, input_board

        for new_predict in new_predicts:
            tmp_board = np.array(cur_board).reshape((15, 15))
            j = 0
            while True:
                # 상대 턴
                if j == 0:
                    y, x = new_predict
                else:
                    input_board = np.expand_dims(tmp_board, axis=(0, -1)).astype(np.float32)
                    if np.random.random() < 0.25:
                        output = model_1.predict(input_board, verbose=0).squeeze()
                    elif np.random.random() < 0.5:
                        output = model_2.predict(input_board, verbose=0).squeeze()
                    elif np.random.random() < 0.75:
                        output = model_3.predict(input_board, verbose=0).squeeze()
                    else:
                        output = model_4.predict(input_board, verbose=0).squeeze()
                    output = output.reshape((15, 15))

                    output = remove_nonzero(output, tmp_board)
                    if np.random.random() > 0.4 and turn == -1:
                        y, x = np.unravel_index(np.argsort(output, axis=None)[-2], output.shape)
                    else:
                        y, x = np.unravel_index(np.argmax(output), output.shape)
                    del output, input_board
                    # print("상대", y, x)
                # 상대가 이김
                if isWin(x, y, tmp_board, -turn):
                    tmp_board[y][x] = -turn
                    lose_scores[i] += 1
                    break
                
                tmp_board[y][x] = -turn

                # 내 턴
                input_board = np.expand_dims(tmp_board, axis=(0, -1)).astype(np.float32)
                output = model_1.predict(input_board, verbose=0).squeeze()
                output = output.reshape((15, 15))

                output = remove_nonzero(output, tmp_board)

                # 상대가 둔 위치에 대해 새로운 예측 -- 내 턴
                y, x = np.unravel_index(np.argmax(output), output.shape)
                # print("나", y, x)

                if np.all(output == 0):
                    print(sum(tmp_board[i][j] == 1 for i in range(15) for j in range(15)))
                    print(sum(tmp_board[i][j] == -1 for i in range(15) for j in range(15)))
                    print("Warning: Model output is all zeros.")
                    print(tmp_board)
                    print(cur_board)

                    input_board = np.expand_dims(tmp_board, axis=(0, -1)).astype(np.float32)
                    output = model_1.predict(input_board, verbose=0).squeeze()
                    output = output.reshape((15, 15))

                    output = remove_nonzero(output, tmp_board)

                    # 모두 0일 경우 새로운 예측 -- 내 턴
                    y, x = np.unravel_index(np.argmax(output), output.shape)
                    print(np.max(output))
                    print("나", y, x)
                
                del output, input_board

                # 내가 이김
                if isWin(x, y, tmp_board, turn):
                    tmp_board[y][x] = turn
                    win_scores[i] += 1
                    break

                tmp_board[y][x] = turn

                # 빈 곳이 10개 미만이면 종료
                cnt = sum(tmp_board[i][j] == 0 for i in range(15) for j in range(15))
                # print(cnt)
                if cnt < 10:
                    break
                j += 1

            # print(tmp_board)

    print(predicts)
    print(win_scores)
    print(lose_scores)
    # 승률 계산
    win_rates = []
    for i in range(len(predicts)):
        if lose_scores[i] == 0:
            win_rate = 1.0 
        else:
            win_rate = win_scores[i] / length[i]

        win_rates.append(win_rate)
    print(win_rates)

    # 승률이 가장 높은 위치 선택
    if np.max(win_rates) <= 0.35:
        return predicts[0]
    
    best_index = None
    best_win_score = -1
    best_lose_score = float('inf')
    max_win_rate = np.max(win_rates)

    for i in range(len(win_rates)):
        if win_rates[i] == max_win_rate:
            if win_scores[i] > best_win_score or (win_scores[i] == best_win_score and lose_scores[i] < best_lose_score):
                best_win_score = win_scores[i]
                best_lose_score = lose_scores[i]
                best_index = i

    return predicts[best_index]