from keras.models import load_model, model_from_json
from keras.models import Model, Sequential
from keras.layers import LSTM, Dense, Input, TimeDistributed, Dropout
from keras.utils import to_categorical
from module import LoadBoard, LoadSequence, OpenPickle
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pickle
from tqdm import tqdm

def LoadModel() -> Model:
    with open('model/hard.json', 'r') as file:
        modelJson = file.read()
    model: Model = model_from_json(modelJson)
    model.load_weights('model/hard.h5')
    return model

def ExtractEncoder(model: Model) -> Model:
    cnnEncoder = Model(inputs=model.input, outputs=model.layers[-2].output)
    cnnEncoder.trainable = False

    return cnnEncoder

def encode_board(encoder, board):
    return np.squeeze(encoder.predict(np.expand_dims(board, axis=(0, -1)).astype(np.float32), verbose=0))

def encode_boards_batch(encoder, boards):
    boards_array = np.array(boards).astype(np.float32)
    boards_array = np.expand_dims(boards_array, axis=-1)
    predictions = encoder.predict(boards_array, verbose=0)
    return [np.squeeze(pred) for pred in predictions]

def preprocess_and_save_embeddings_with_labels(games: list[list[list[int]]], encoder: Model, seq_length: int, save_path="data3.pkl"):
    embeddings = []
    labels = []

    for game in tqdm(games, desc="Processing games for embeddings and labels"):
        game_embeddings = []
        game_labels = []

        previous_boards = []
        current_boards = []
        move_positions = []

        # 새로운 돌이 놓인 위치와 해당 보드들을 리스트에 저장
        for i in range(len(game) - 1):
            previous_board = np.array(game[i])
            current_board = np.array(game[i+1])

            # 새로운 돌이 놓인 위치 찾기
            move_position = np.where((previous_board.flatten() == 0) &
                                     (current_board.flatten() != previous_board.flatten()))[0]

            if len(move_position) == 0:
                continue  # 새로운 돌이 없는 경우 다음으로 넘어감

            # 현재 보드를 리스트에 추가
            previous_boards.append(previous_board)
            move_positions.append(move_position[0])

        if len(previous_boards) == 0:
            continue  # 유효한 데이터가 없는 경우 다음 게임으로

        # 이전 보드들에 대해 한 번에 임베딩 생성
        previous_boards_array = np.array(previous_boards).astype(np.float32)
        previous_boards_array = np.expand_dims(previous_boards_array, axis=-1)
        embeddings_batch = encoder.predict(previous_boards_array, verbose=0)
        embeddings_batch = [np.squeeze(emb) for emb in embeddings_batch]

        # 임베딩과 라벨을 각각 저장
        game_embeddings.extend(embeddings_batch)
        game_labels.extend([to_categorical(pos, num_classes=225) for pos in move_positions])

        # 시퀀스 길이 적용하여 데이터 생성
        for j in range(len(game_embeddings) - seq_length + 1):
            embeddings.append(game_embeddings[j:j+seq_length])
            labels.append(game_labels[j+seq_length-1])

    # Pickle로 임베딩과 라벨 저장
    with open(save_path, 'wb') as f:
        pickle.dump({'embeddings': embeddings, 'labels': labels}, f)



def load_embeddings_and_labels(filepath="data3.pkl"):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    embeddings = data['embeddings']  # 리스트 형태
    labels = data['labels']          # 리스트 형태

    # numpy 배열로 변환
    x = np.array(embeddings)
    y = np.array(labels)

    return x, y

def CreateLSTMSequenceModel(encoder: Model, seq_length: int) -> Model:
    embedding_dim = encoder.output_shape[1]

    model = Sequential([
        # LSTM 레이어
        LSTM(64, input_shape=(seq_length, embedding_dim), return_sequences=True),
        Dropout(0.3),  # 과적합 방지를 위한 Dropout
        LSTM(32, return_sequences=True),
        Dropout(0.3),
        LSTM(16),  # 최종 LSTM 레이어로 return_sequences=False

        # Dense 레이어로 돌 위치 예측
        Dense(128, activation='relu'),  # 추가적인 Dense 레이어로 패턴 추출
        Dropout(0.3),
        Dense(225, activation='softmax')  # 15x15 보드의 각 위치에 대한 확률 분포
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

SEQ_LENGTH = 10

if __name__ == "__main__":
    print("Loading model...")
    model = LoadModel()
    encoder = ExtractEncoder(model)

    boards = []
    print("Loading sequence...")
    boards = OpenPickle("boards3.pkl")

    print("Creating CNN to LSTM sequence...")
    preprocess_and_save_embeddings_with_labels(boards, encoder, SEQ_LENGTH)

    x, y = load_embeddings_and_labels()
    print(x.shape, y.shape)



    # print("Creating LSTM sequence model...")
    # model = CreateLSTMSequenceModel(encoder, SEQ_LENGTH)

    # model.fit(x, y, epochs=100, batch_size=32)

    # print(model.predict(x[:1]))

