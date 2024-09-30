# -*- coding: utf-8 -*-
# pip install pipeline
"""### ealvaradob/bert-finetuned-phishing"""

"""
모델은 pytorch만 지원
"""

# Use a pipeline as a high-level helper
import pandas as pd
import os
import logging
from sklearn.model_selection import train_test_split
from transformers import pipeline,AutoTokenizer, TFAutoModelForSequenceClassification, AutoModelForSequenceClassification
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from tqdm import tqdm
#from tensorflow_directml import load


# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # 디버그 수준으로 로그 출력


# 데이터 경로 설정
ROOT_DIR = "../custom_datasets"
MODEL_DIR = './saved_model'
TOKENIZER_DIR = './saved_tokenizer'

# GPU 사용 가능한지 확인 - ㅇㅋ
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cpu':
    response = input("GPU를 찾을 수 없습니다. CPU로 계속 실행하시겠습니까? (y/n): ").strip().lower()
    if response != 'y':
        print("프로그램을 종료합니다.")
        exit()
    else:
        print("CPU로 실행합니다.")
else:
    print(f"GPU 사용 가능: {torch.cuda.device_count()} GPUs")


"""
def load_and_prepare_data(root_dir, n_samples=10000):
    
    #데이터를 로드하고 전처리하여 훈련 및 검증 세트를 반환합니다.
    
    # 데이터 로드
    logger.info("피싱 데이터 로드 중...")
    phishing_df = pd.read_csv(
        os.path.join(root_dir, 'combined_phishing_data.txt'),
        header=None,
        delimiter='\t',
        names=['label', 'URL']
    )
    logger.info(f"피싱 데이터 로드 완료: {len(phishing_df)}개 샘플")

    logger.info("정상 데이터 로드 중...")
    benign_df = pd.read_csv(
        os.path.join(root_dir, 'combined_safe_data.txt'),
        header=None,
        delimiter='\t',
        names=['label', 'URL']
    )
    logger.info(f"정상 데이터 로드 완료: {len(benign_df)}개 샘플")

    # 각 클래스에서 일부 샘플 추출
    logger.info(f"피싱 데이터에서 {n_samples}개 샘플 추출 중...")
    phishing_df = phishing_df.sample(n=n_samples, random_state=42)
    logger.info(f"추출된 피싱 샘플 수: {len(phishing_df)}")

    logger.info(f"정상 데이터에서 {n_samples}개 샘플 추출 중...")
    benign_df = benign_df.sample(n=n_samples, random_state=42)
    logger.info(f"추출된 정상 샘플 수: {len(benign_df)}")

    # 데이터 결합
    logger.info("피싱 및 정상 데이터 결합 중...")
    df = pd.concat([phishing_df, benign_df], ignore_index=True)
    logger.info(f"결합된 데이터셋 크기: {len(df)}")

    # 레이블 매핑 (-1: 피싱 ⇒ 1, +1: 정상 ⇒ 0)
    logger.info("레이블 매핑 중...")
    label_mapping = {-1: 1, +1: 0}
    df['label'] = df['label'].map(label_mapping)

    # 데이터 셔플링
    logger.info("데이터 셔플링 중...")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # 데이터 통계 출력
    logger.info(f"레이블 분포:\n{df['label'].value_counts()}")

    # 텍스트와 레이블 추출
    texts = df['URL'].tolist()
    labels = df['label'].tolist()

    # 훈련 세트와 검증 세트로 분할
    logger.info("훈련 세트와 검증 세트로 분할 중...")
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    logger.info(f"훈련 샘플 수: {len(train_texts)}")
    logger.info(f"검증 샘플 수: {len(val_texts)}")

    return train_texts, val_texts, train_labels, val_labels
"""

def load_and_prepare_data(root_dir, n_samples=5000):
    """
    데이터를 로드하고 전처리하여 훈련 및 검증 세트를 반환합니다.
    """
    # 데이터 로드
    logger.info("피싱 데이터 로드 중...")
    phishing_df = pd.read_csv(
        os.path.join(root_dir, 'combined_phishing_data.txt'),
        header=None,
        delimiter='\t',
        names=['label', 'URL']
    )
    logger.info(f"피싱 데이터 로드 완료: {len(phishing_df)}개 샘플")

    logger.info("정상 데이터 로드 중...")
    benign_df = pd.read_csv(
        os.path.join(root_dir, 'combined_safe_data.txt'),
        header=None,
        delimiter='\t',
        names=['label', 'URL']
    )
    logger.info(f"정상 데이터 로드 완료: {len(benign_df)}개 샘플")

    # 각 클래스에서 일부 샘플 추출
    phishing_df = phishing_df.sample(n=n_samples, random_state=42)
    benign_df = benign_df.sample(n=n_samples, random_state=42)

    # 데이터 결합
    df = pd.concat([phishing_df, benign_df], ignore_index=True)

    # 레이블 매핑 (-1: 피싱 ⇒ 1, +1: 정상 ⇒ 0)
    label_mapping = {-1: 1, +1: 0}
    df['label'] = df['label'].map(label_mapping)

    # 데이터 셔플링
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # 텍스트와 레이블 추출
    texts = df['URL'].tolist()
    labels = df['label'].tolist()

    # 훈련 세트와 검증 세트로 분할
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    return train_texts, val_texts, train_labels, val_labels
"""
def train_model(train_texts, train_labels, val_texts, val_labels, epochs=3, batch_size=16, model=None):
    
    #모델을 훈련하고 검증 세트 성능을 출력합니다.
    
    # 토크나이저 로드
    logger.info("토크나이저 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained("ealvaradob/bert-finetuned-phishing")
    logger.info("토크나이저 로드 완료")

    # 데이터 토큰화
    logger.info("훈련 데이터 토큰화 중...")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    logger.info("검증 데이터 토큰화 중...")
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)

    # TensorFlow 데이터셋 생성
    logger.info("TensorFlow 데이터셋 생성 중...")
    train_dataset = tf.data.Dataset.from_tensor_slices((
        dict(train_encodings),
        train_labels
    ))
    val_dataset = tf.data.Dataset.from_tensor_slices((
        dict(val_encodings),
        val_labels
    ))
    logger.info("TensorFlow 데이터셋 생성 완료")

    # 모델 로드 또는 초기화
    if model is None:
        logger.info("모델 로드 중...")
        # PyTorch 모델 로드
        model = AutoModelForSequenceClassification.from_pretrained("ealvaradob/bert-finetuned-phishing")
        #model = TFAutoModelForSequenceClassification.from_pretrained("ealvaradob/bert-finetuned-phishing", from_pt=True)
        logger.info("모델 로드 완료")
    else:
        logger.info("기존 모델을 사용하여 훈련을 계속합니다.")

    # 모델 컴파일
    logger.info("모델 컴파일 중...")
    optimizer = Adam(learning_rate=3e-5)
    model.compile(optimizer=optimizer, metrics=['accuracy'])
    logger.info("모델 컴파일 완료")

    # 데이터셋 배치 및 셔플
    logger.info("데이터셋 배치 및 셔플 중...")
    train_dataset = train_dataset.shuffle(len(train_dataset), seed=42).batch(batch_size)
    val_dataset = val_dataset.batch(batch_size)
    logger.info("데이터셋 준비 완료")

    # 모델 훈련
    logger.info("모델 훈련 시작")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs
    )
    logger.info("모델 훈련 완료")

    # 검증 세트 평가
    logger.info("검증 세트 평가 중...")
    loss, accuracy = model.evaluate(val_dataset)
    logger.info(f"검증 세트 손실: {loss:.4f}")
    logger.info(f"검증 세트 정확도: {accuracy * 100:.2f}%")

    # 모델 저장
    logger.info("모델 저장 중...")
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(TOKENIZER_DIR)
    logger.info("모델 저장 완료")

    return model, tokenizer
"""

def train_model(train_texts, train_labels, val_texts, val_labels, epochs=3, batch_size=4, model=None):
    """
    모델을 훈련하고 검증 세트 성능을 출력합니다.
    """
    # 토크나이저 로드
    logger.info("토크나이저 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained("ealvaradob/bert-finetuned-phishing")
    logger.info("토크나이저 로드 완료")

    # 데이터 토큰화
    logger.info("훈련 데이터 토큰화 중...")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors='pt')
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, return_tensors='pt')

    # PyTorch 데이터셋 생성
    #train_dataset = TensorDataset(train_encodings['input_ids'], torch.tensor(train_labels))
    #val_dataset = TensorDataset(val_encodings['input_ids'], torch.tensor(val_labels))

    # PyTorch 데이터셋 생성 시 attention_mask 포함
    train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], torch.tensor(train_labels))
    val_dataset = TensorDataset(val_encodings['input_ids'], val_encodings['attention_mask'], torch.tensor(val_labels))

    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # 모델 로드
    logger.info("모델 로드 중...")
    model = AutoModelForSequenceClassification.from_pretrained("ealvaradob/bert-finetuned-phishing")
    model.to(device)
    logger.info("모델 로드 완료")

    # 옵티마이저 설정
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Gradient Accumulation 설정
    accumulation_steps = 4  # 4번의 작은 배치 후에 한 번의 업데이트

    # 모델 학습
    model.train() # 위치??
    for epoch in range(epochs):
        optimizer.zero_grad() # 경사 초기화
        epoch_loss = 0
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")):
            inputs = batch[0].to(device)
            attention_mask = batch[1].to(device)  # attention_mask 추가
            labels = batch[2].to(device)

            # forward pass
            outputs = model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()

            epoch_loss += loss.item()  # 누적 손실 계산

             # 경사 누적 후 optimizer 업데이트
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()  # 경사 초기화

            #optimizer.step()

            # GPU 메모리 캐시 정리
            torch.cuda.empty_cache()

        #logger.info(f"에포크 {epoch+1}/{epochs} - 손실: {loss.item()}")
        logger.info(f"에포크 {epoch+1}/{epochs} - 손실: {epoch_loss / len(train_loader):.4f}")

    # 검증 세트 평가
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch[0].to(device)
            attention_mask = batch[1].to(device)  # attention_mask 추가
            labels = batch[2].to(device)

            # forward pass
            outputs = model(input_ids=inputs, attention_mask=attention_mask)
            
            # 모델의 출력 크기와 labels 크기를 맞추기 위해 batch 크기를 확인합니다.
            if outputs.logits.size(0) != labels.size(0):
                raise ValueError(f"Output size {outputs.logits.size(0)} and labels size {labels.size(0)} don't match.")

            # 예측값
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    logger.info(f"검증 세트 정확도: {accuracy:.2f}%")

     # 모델 저장
    logger.info("모델 저장 중...")
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(TOKENIZER_DIR)
    logger.info("모델 저장 완료")

    return model, tokenizer

"""
def load_model_and_tokenizer(model_dir=MODEL_DIR, tokenizer_dir=TOKENIZER_DIR):
    #저장된 모델과 토크나이저를 로드합니다.
    
    if not os.path.exists(model_dir) or not os.path.exists(tokenizer_dir):
        logger.info("저장된 모델 또는 토크나이저가 존재하지 않습니다.")
        return None, None
    else:
        logger.info("저장된 모델과 토크나이저 로드 중...")
        model = TFAutoModelForSequenceClassification.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        logger.info("모델과 토크나이저 로드 완료")
        return model, tokenizer
"""

def load_model_and_tokenizer(model_dir=MODEL_DIR, tokenizer_dir=TOKENIZER_DIR):
    """
    저장된 모델과 토크나이저를 로드합니다. PyTorch 기반으로 로드합니다.
    """
    if not os.path.exists(model_dir) or not os.path.exists(tokenizer_dir):
        logger.info("저장된 모델 또는 토크나이저가 존재하지 않습니다.")
        return None, None
    else:
        logger.info("저장된 모델과 토크나이저 로드 중...")
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)  # PyTorch 모델 로드
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        logger.info("모델과 토크나이저 로드 완료")
        return model, tokenizer

"""
def predict_url(model, tokenizer):
    
    #사용자로부터 URL을 입력받아 피싱 여부를 예측합니다.
    
    while True:
        user_input = input("피싱 여부를 확인할 URL을 입력하세요 (종료하려면 'exit' 입력): ")
        if user_input.lower() == 'exit':
            logger.info("프로그램을 종료합니다.")
            break

        # 새로운 URL 리스트
        new_urls = [user_input]

        logger.info("새로운 데이터 토큰화 중...")
        # 토크나이저에 입력
        new_encodings = tokenizer(new_urls, truncation=True, padding=True, return_tensors='tf')

        logger.info("예측 수행 중...")
        # 예측
        predictions = model(new_encodings)
        predicted_probs = tf.nn.softmax(predictions.logits, axis=-1).numpy()
        predicted_label = tf.argmax(predictions.logits, axis=1).numpy()[0]
        phishing_prob = predicted_probs[0][1] * 100  # 피싱일 확률 (%)

        # 레이블 디코딩 (1 ⇒ 피싱, 0 ⇒ 정상)
        label_decoding = {1: '피싱', 0: '정상'}
        decoded_label = label_decoding[predicted_label]

        # 결과 출력
        logger.info(f"URL: {user_input}")
        logger.info(f"예측 레이블: {decoded_label}")
        logger.info(f"피싱 확률: {phishing_prob:.2f}%")
"""

def predict_url(model, tokenizer):
    """
    사용자로부터 URL을 입력받아 피싱 여부를 예측합니다.
    """
    model.eval()
    while True:
        user_input = input("피싱 여부를 확인할 URL을 입력하세요 (종료하려면 'exit' 입력): ")
        if user_input.lower() == 'exit':
            logger.info("프로그램을 종료합니다.")
            break

        new_urls = [user_input]
        encodings = tokenizer(new_urls, truncation=True, padding=True, return_tensors='pt')
        inputs = encodings['input_ids'].to(device)

        with torch.no_grad():
            outputs = model(inputs)
            predicted_probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_label = torch.argmax(predicted_probs, dim=1).item()
            phishing_prob = predicted_probs[0][1].item() * 100

        label_decoding = {1: '피싱', 0: '정상'}
        decoded_label = label_decoding[predicted_label]

        logger.info(f"URL: {user_input}")
        logger.info(f"예측 레이블: {decoded_label}")
        logger.info(f"피싱 확률: {phishing_prob:.2f}%")

def main():
    while True:
        user_choice = input("작업을 선택하세요 ('start', 'run', 'train', 'exit'): ").strip().lower()

        if user_choice == 'start':
            # 데이터 로드 및 준비
            train_texts, val_texts, train_labels, val_labels = load_and_prepare_data(ROOT_DIR)

            # 모델 훈련 및 평가
            model, tokenizer = train_model(train_texts, train_labels, val_texts, val_labels)

            # URL 예측
            predict_url(model, tokenizer)

        elif user_choice == 'run':
            # 저장된 모델과 토크나이저 로드
            model, tokenizer = load_model_and_tokenizer()

            if model is None or tokenizer is None:
                print("저장된 모델이 없어서 모델을 새로 훈련시킵니다.")
                # 데이터 로드 및 준비
                train_texts, val_texts, train_labels, val_labels = load_and_prepare_data(ROOT_DIR)
                # 모델 훈련 및 평가
                model, tokenizer = train_model(train_texts, train_labels, val_texts, val_labels)
                # URL 예측
                predict_url(model, tokenizer)
            else:
                # URL 예측
                predict_url(model, tokenizer)

        elif user_choice == 'train':
            # 저장된 모델과 토크나이저 로드
            model, tokenizer = load_model_and_tokenizer()

            if model is None or tokenizer is None:
                print("저장된 모델이 없어서 새로운 모델을 훈련합니다.")
                # 데이터 로드 및 준비
                train_texts, val_texts, train_labels, val_labels = load_and_prepare_data(ROOT_DIR)
                # 모델 훈련 및 평가
                model, tokenizer = train_model(train_texts, train_labels, val_texts, val_labels)
                # URL 예측
                predict_url(model, tokenizer)
            else:
                # 데이터 로드 및 준비
                train_texts, val_texts, train_labels, val_labels = load_and_prepare_data(ROOT_DIR)
                # 기존 모델로 추가 훈련
                model, tokenizer = train_model(train_texts, train_labels, val_texts, val_labels, model=model)
                # URL 예측
                predict_url(model, tokenizer)

        elif user_choice == 'exit':
            print("프로그램을 종료합니다.")
            break

        else:
            print("올바르지 않은 입력입니다. 다시 시도해주세요.")

if __name__ == "__main__":
    main()
    
"""
pipe = pipeline("text-classification", model="ealvaradob/bert-finetuned-phishing") #pipeline 생성 후 task 지정 & 입력 넣기 

# 레포에 저장된 파일 경로
ROOT_DIR = "../custom_datasets"

## 피싱 데이터

# 텍스트 파일을 DataFrame으로 로드
df = pd.read_csv(os.path.join(ROOT_DIR, 'result/combined_phishing_data.txt'), header=None, delimiter='\t', names=['label', 'URL'])

# 상위 100개의 행만 처리
df = df.head(100)

# 'label' 컬럼 이름을 '실제 피싱 여부'로 변경
df.rename(columns={'label': 'Actual'}, inplace=True)

# URL 리스트 생성
urls = df['URL'].tolist()

# Hugging Face 모델로 예측 수행
results = pipe(urls)

# 예측된 'label' (phishing 여부)을 새로운 컬럼에 추가
df['Predicted Label'] = [result['label'] for result in results]

# 예측 결과를 DataFrame에 추가 ('score'는 피싱일 확률)
df['Prediction Score'] = [f"{result['score'] * 100:.2f} %" for result in results]

# DataFrame을 CSV로 저장
df.to_csv(os.path.join(ROOT_DIR, 'combined_phishing_data_with_results_bert.csv'), index=False)

## 정상 데이터

# 텍스트 파일을 DataFrame으로 로드
df = pd.read_csv(os.path.join(ROOT_DIR, 'combined_benign_data.txt'), header=None, delimiter='\t', names=['label', 'URL'])

# 상위 100개의 행만 처리
df = df.head(100)

# 'label' 컬럼 이름을 '실제 피싱 여부'로 변경
df.rename(columns={'label': 'Actual'}, inplace=True)

# URL 리스트 생성
urls = df['URL'].tolist()

# Hugging Face 모델로 예측 수행
results = pipe(urls)

# 예측된 'label' (phishing 여부)을 새로운 컬럼에 추가
df['Predicted Label'] = [result['label'] for result in results]

# 예측 결과를 DataFrame에 추가 ('score'는 피싱일 확률)
df['Prediction Score'] = [f"{result['score'] * 100:.2f} %" for result in results]

# DataFrame을 CSV로 저장
df.to_csv(os.path.join(ROOT_DIR, 'result/combined_benign_data_with_results_bert.csv'), index=False)

"""