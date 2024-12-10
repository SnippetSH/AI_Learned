import datasets
import torch
import torch.nn as nn
import random

from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR, LambdaLR, CosineAnnealingLR
from torch.amp import autocast, GradScaler
from torch import Tensor
from tqdm import tqdm

BATCH_SIZE = 32
DROPOUT_RATE = 0.2
EPOCHS = 8
MAX_LEN = 80
D_MODEL = 128
LEARNING_RATE = 1e-3

TEST_DATA = [
    "it’s a charming and often affecting journey.",
    "Unflinchingly bleak and desperate.",
    "allows us to hope that nolan is poised to embark a major career as a commercial yet inventive filmmaker",
    "the acting, costumes, music, cinematography and sound are all astounding given the production’s austere locales.",
    "it’s slow – very, very slow.",
    "although laced with humor and a few fanciful touches, the film is a refreshingly serious look at young women.",
    "a sometimes tedious film.",
    "or doing last year’s taxes with your ex-wife.",
    "you don’t have to know about music to appreciate the film’s easygoing blend of comedy and romance.",
    "in exactly 89 minutes, most of which passed as slowly as if i’d been sitting naked on an igloo, formula 51 sank from quirky to jerky to utter turkey."
]

# 데이터셋 로드 및 전처리
def load():
    ds = datasets.load_dataset("stanfordnlp/sst2")
    return ds

def build_vocab(ds) -> dict[str, int]:
    counter = Counter()
    for ex in ds["train"]:
        counter.update(ex["sentence"].split())

    vocab = {word: idx for idx, (word, _) in enumerate(counter.items(), start=1)}
    vocab["<PAD>"] = 0
    return vocab

def encode_sentence(sentence, vocab, max_len=MAX_LEN) -> list[int]:
    tokens = [vocab.get(word, 0) for word in sentence.split()]
    if len(tokens) > max_len:
        tokens = tokens[:max_len]
    else:
        tokens += [0] * (max_len - len(tokens))
    
    return tokens

# ngram을 사용한 전처리
# 사용하지 않았을 때 성능이 더 좋았음
def encode_sentence_with_ngram(sentence, vocab, max_len=MAX_LEN, n=5) -> list[int]:
    def ngrams(word, n):
        return [word[i:i+n] for i in range(len(word)-n+1)]

    tokens = []
    for word in sentence.split():
        ngram_tokens = [vocab.get(ngram, 0) for ngram in ngrams(word, n)]
        tokens.extend(ngram_tokens)

    if len(tokens) > max_len:
        tokens = tokens[:max_len]
    else:
        tokens += [0] * (max_len - len(tokens))

    return tokens

# 데이터셋 클래스
class SST2Dataset(Dataset):
    def __init__(self, ds, vocab, max_len=MAX_LEN):
        self.ds = ds
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, index) -> tuple[Tensor, Tensor]:
        sentence = self.ds[index]["sentence"]
        label = self.ds[index]["label"]

        input_ids = torch.tensor(encode_sentence(sentence, self.vocab, self.max_len))
        attention_mask = (input_ids != 0).float()
        # label = torch.tensor(label, dtype=torch.float) # BCEWithLogitsLoss를 사용할 경우 
        label = torch.tensor(label, dtype=torch.long) # CrossEntropyLoss를 사용할 경우
        
        return input_ids, label, attention_mask

# 트랜스포머와 CrossEntropyLoss를 사용한 이진 분류 모델
"""
BCE와 CrossEntropy 모두 사용해본 결과
CrossEntropyLoss가 더 좋은 성능을 보임

또한, 첫 토큰의 출력을 대표 벡터로 사용하는 것보다 
평균값을 사용하는 것이 나았음
"""
class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=D_MODEL, n_heads=4, num_layers=4, dim_feedforward=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward, dropout=DROPOUT_RATE, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)

        self.classifier = nn.Linear(d_model, 2) # CrossEntropyLoss를 사용할 경우 
        # self.classifier = nn.Linear(d_model, 1) # BCEWithLogitsLoss를 사용할 경우

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids) # shape = (batch, seq_len, d_model)

        src_key_padding_mask = (input_ids == 0) # 또는 attention_mask 사용 가능
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        # 첫 토큰의 출력을 대표 벡터로 사용
        # x = x[:, 0, :]
        # logits = self.classifier(x).squeeze(-1) # (batch, 1) -> (batch,)

        # 또는 x = x.mean(dim=1) 사용 (평균값)
        x = x.mean(dim=1)
        logits = self.classifier(x)
        
        return logits
    
# Linear Warmup & Decay를 사용하는 스케줄러
# 이 스케줄러는 사용하지 않고 단순 StepLR을 사용했음
def get_linear_warmup_scheduler(optimizer, warmup_steps, training_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(training_steps - current_step) / float(max(1, training_steps - warmup_steps)))
    return LambdaLR(optimizer, lr_lambda)

# Colab을 통해 학습했기 때문에, GPU를 사용하여 병렬로 학습을 진행함
def main():
    ds = load()
    vocab = build_vocab(ds)

    # Create a dataset and dataloader
    train_ds = SST2Dataset(ds["train"], vocab)
    val_ds = SST2Dataset(ds["validation"], vocab)

    input_ids, label, attention_mask = train_ds[0]
    print("Example input_ids:", input_ids)
    print("Example label:", label)
    print("Example attention_mask:", attention_mask)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentimentClassifier(len(vocab)).to(device)
    # model = torch.compile(model)

    # criterion = nn.BCEWithLogitsLoss() # BCEWithLogitsLoss를 사용할 경우
    criterion = nn.CrossEntropyLoss() # CrossEntropyLoss를 사용할 경우 
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5) # lr: learning rate, default=1e-3

    # StepLR: 일정 단계마다 lr 감소
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    # Linear Warmup & Decay 
    # total_steps = len(train_loader) * EPOCHS
    # warmup_steps = int(0.1 * total_steps)
    # scheduler = get_linear_warmup_scheduler(optimizer, warmup_steps, total_steps)

    # CosineAnnealingLR 
    # scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Early Stopping 기반 변수
    early_stopping_patience = 3
    best_val_accuracy = 0.0
    best_val_loss = float('inf')
    patience_counter_acc = 0
    patience_counter_loss = 0

    scaler = GradScaler()
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        train_progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", leave=False)
        for batch in train_progress:
            input_ids, labels, attention_mask = batch

            # CUDA로 데이터를 전송
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            attention_mask = attention_mask.to(device)

            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                logits = model(input_ids, attention_mask=attention_mask)

                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        val_progress = tqdm(val_loader, desc="Validation", leave=False)
        with torch.no_grad():
            for batch in val_progress:
                input_ids, labels, attention_mask = batch

                # CUDA로 데이터를 전송
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                attention_mask = attention_mask.to(device)

                logits = model(input_ids, attention_mask=attention_mask)
                loss = criterion(logits, labels)
                val_loss += loss.item()

                preds = torch.argmax(logits, dim=1) # CrossEntropyLoss를 사용할 경우
                # preds = (logits > 0).long() # BCEWithLogitsLoss를 사용할 경우
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_accuracy = correct / total
        val_loss /= len(val_loader)

        print(f"Epoch {epoch + 1}/{EPOCHS}, Train Loss: {total_loss / len(train_loader):.4f}, Val Accuracy: {val_accuracy:.4f}, Val Loss: {val_loss}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            patience_counter_acc = 0
        else:
            patience_counter_acc += 1
            print(f"Early stopping counter (Accuracy): {patience_counter_acc}/{early_stopping_patience}")
            if patience_counter_acc >= early_stopping_patience:
                print("Early stopping triggered (Accuracy).")
                break

        # Early Stopping for Loss
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     patience_counter_loss = 0
        # else:
        #     patience_counter_loss += 1
        #     print(f"Early stopping counter (Loss): {patience_counter_loss}/{early_stopping_patience}")
        #     if patience_counter_loss >= early_stopping_patience:
        #         print("Early stopping triggered due to no improvement in validation loss.")
        #         break

    return model, vocab

def preprocess_test_data(vocab, max_len=MAX_LEN):
    input_ids = []
    attention_masks = []

    for sentence in TEST_DATA:
        ids = torch.tensor(encode_sentence(sentence, vocab, max_len))
        mask = (ids != 0).float()
        input_ids.append(ids)
        attention_masks.append(mask)

    input_ids = torch.stack(input_ids)
    attention_masks = torch.stack(attention_masks)

    return input_ids, attention_masks

def test_model(model, vocab):
    model.eval()
    device = next(model.parameters()).device  

    input_ids, attention_masks = preprocess_test_data(vocab, max_len=MAX_LEN)

    # CUDA로 데이터를 전송
    input_ids = input_ids.to(device)
    attention_masks = attention_masks.to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_masks)
        preds = torch.argmax(logits, dim=1) # CrossEntropyLoss를 사용할 경우
        # preds = (logits > 0).long() # BCEWithLogitsLoss를 사용할 경우

    for i, sentence in enumerate(TEST_DATA):
        print(f"Sentence: {sentence}")
        print(f"Prediction: {'Positive' if preds[i] == 1 else 'Negative'}")
        print()

## use CrossEntropyLoss
"""
Sentence: it’s a charming and often affecting journey.
Prediction: Positive

Sentence: Unflinchingly bleak and desperate.
Prediction: Negative

Sentence: allows us to hope that nolan is poised to embark a major career as a commercial yet inventive filmmaker
Prediction: Positive

Sentence: the acting, costumes, music, cinematography and sound are all astounding given the production’s austere locales.
Prediction: Positive

Sentence: it’s slow – very, very slow.
Prediction: Negative

Sentence: although laced with humor and a few fanciful touches, the film is a refreshingly serious look at young women.
Prediction: Positive

Sentence: a sometimes tedious film.
Prediction: Negative

Sentence: or doing last year’s taxes with your ex-wife.
Prediction: Negative

Sentence: you don’t have to know about music to appreciate the film’s easygoing blend of comedy and romance.
Prediction: Positive

Sentence: in exactly 89 minutes, most of which passed as slowly as if i’d been sitting naked on an igloo, formula 51 sank from quirky to jerky to utter turkey.
Prediction: Negative

"""
if __name__ == "__main__":
    model, vocab = main()
    test_model(model, vocab)