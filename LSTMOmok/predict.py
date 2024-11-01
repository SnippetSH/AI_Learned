from keras.models import model_from_json
import numpy as np

def remove_nonzero(output, board):
    for i in range(15):
        for j in range(15):
            if board[i][j] != 0:
                output[i][j] = 0

    return output

# CNN 모델로 보드 상태를 임베딩 벡터로 변환하는 함수
def get_cnn_embedding(cnn_model, board_state):
    board_state = np.expand_dims(board_state, axis=(0, -1))  # (1, 15, 15, 1) 형태로 변환
    embedding = cnn_model.predict(board_state)
    return np.squeeze(embedding)  # (embedding_dim,) 형태

def predict_before_five_moves(model, board):
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
    output = model.predict(input_board, verbose=0).squeeze()
    output = output.reshape((15, 15))

    output = remove_nonzero(output, board)
    y, x = np.unravel_index(np.argmax(output), output.shape)

    return x, y

# 전체 예측 프로세스
def predict_next_move(cnn_model, lstm_model, board_sequence, seq_length=10, min_moves=5):
    if len(board_sequence) < min_moves:
        raise ValueError(f"At least {min_moves} moves are required for a meaningful prediction.")

    # 임베딩 벡터 시퀀스 준비
    embeddings = []
    # 최근 seq_length 수만 사용하여 임베딩 시퀀스 생성
    for board in board_sequence[-seq_length:]:
        embedding = get_cnn_embedding(cnn_model, board)
        embeddings.append(embedding)
    
    # 임베딩 시퀀스를 LSTM 입력 형태로 변환 (1, seq_length, embedding_dim)
    embeddings = np.array(embeddings).reshape((1, seq_length, -1))
    embeddings = embeddings.reshape((embeddings.shape[0], embeddings.shape[1], 15, 15, 1))
    
    # LSTM 모델을 통해 다음 수 예측
    next_move_prob = lstm_model.predict(embeddings).squeeze()  # (1, 225) 형태의 확률 분포 출력
    next_move_prob = next_move_prob.reshape((15, 15))               # 가장 높은 확률의 위치 선택
    
    y, x = np.unravel_index(np.argmax(next_move_prob), next_move_prob.shape)
    return x, y

def main():
    with open('model/hard.json', 'r') as file:
        cnn_model_json = file.read()
    cnn_model = model_from_json(cnn_model_json)
    cnn_model.load_weights('model/hard.h5')

    # cnn_model.summary()
    

    with open('model/lstm_model_structure_acc(63)_valacc(26).json', 'r') as file:
        lstm_model_json = file.read()
    lstm_model = model_from_json(lstm_model_json)
    lstm_model.load_weights('model/lstm_model_weights (2).h5')


    board_sequence = []
    board = [[0 for _ in range(15)] for _ in range(15)]
    board_sequence.append(board)

    cnt = 0
    while True:
        if cnt < 5:
            x, y = predict_before_five_moves(cnn_model, board)
            board[y][x] = 1
            board_sequence.append(board)
            cnt += 1
        else:
            x, y = predict_next_move(cnn_model, lstm_model, board_sequence)
            board[y][x] = 1
            board_sequence.append(board)
            print("으악")

        print_board = [[0 for _ in range(15)] for _ in range(15)]
        for i in range(15):
            for j in range(15):
                if board[i][j] == -1:
                    print_board[i][j] = "2"
                else:
                    print_board[i][j] = str(board[i][j])


        print("0 1 2 3 4 5 6 7 8 9 0 1 2 3 4")
        i = 0
        for row in print_board:
            print(" ".join(str(cell) for cell in row), end="")
            print(f" :{i}")
            i += 1
        print()

        x, y = map(int, input("input x, y: ").split())
        board[y][x] = -1
        board_sequence.append(board)

if __name__ == "__main__":
    main()

