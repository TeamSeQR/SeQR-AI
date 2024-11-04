# -*- coding: utf-8 -*-
# pip install pipeline
"""### ealvaradob/bert-finetuned-phishing"""

"""
모델은 pytorch만 지원
"""

# Use a pipeline as a high-level helper
import pandas as pd
import torch.nn.functional as F
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
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
#from tensorflow_directml import load


# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # 디버그 수준으로 로그 출력


# 데이터 경로 설정
ROOT_DIR = "../custom_datasets"
MODEL_DIR = './saved_model'
TOKENIZER_DIR = './saved_tokenizer'
RESULT_DIR = "../result_datasets"

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

def load_and_prepare_data(root_dir, n_samples=500):
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

    # 결측값 제거
    df = df.dropna(subset=['URL'])

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

def load_and_add_data(root_dir, n_samples=25):
    """
    데이터를 로드하고 전처리하여 훈련 세트를 반환합니다.
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
        os.path.join(root_dir, 'evaluation_safe_data.txt'),
        header=None,
        delimiter='\t',
        names=['label', 'URL']
    )
    logger.info(f"정상 데이터 로드 완료: {len(benign_df)}개 샘플")

    # 정상 데이터는 전체 사용 (정상 데이터가 25개인 것을 가정)
    logger.info("정상 데이터 전체 사용")
    benign_sampled_df = benign_df

    # 피싱 데이터는 25개만 무작위로 추출
    logger.info("피싱 데이터에서 25개 샘플 추출 중...")
    phishing_sampled_df = phishing_df.sample(n=n_samples, random_state=42)
    logger.info(f"추출된 피싱 샘플 수: {len(phishing_sampled_df)}")

    # 데이터 결합
    df = pd.concat([phishing_sampled_df, benign_sampled_df], ignore_index=True)
    logger.info(f"결합된 데이터셋 크기: {len(df)}")

    # 레이블 매핑 (-1: 피싱 ⇒ 1, +1: 정상 ⇒ 0)
    label_mapping = {-1: 1, +1: 0}
    df['label'] = df['label'].map(label_mapping)

    # 데이터 셔플링
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 텍스트와 레이블 추출
    texts = df['URL'].tolist()
    labels = df['label'].tolist()

    return texts, labels



def load_validation_data(root_dir):
    """
    검증 데이터를 로드하고 전처리하여 텍스트와 레이블 리스트를 반환합니다.
    """
    # 데이터 로드
    logger.info("검증 피싱 데이터 로드 중...")
    phishing_df = pd.read_csv(
        #os.path.join(root_dir, 'evaluation_phishing_data.txt'),
        os.path.join(root_dir, 'combined_phishing_data.txt'),
        header=None,
        delimiter='\t',
        names=['label', 'URL']
    )
    logger.info(f"검증 피싱 데이터 로드 완료: {len(phishing_df)}개 샘플")

    logger.info("검증 정상 데이터 로드 중...")
    benign_df = pd.read_csv(
        #os.path.join(root_dir, 'evaluation_safe_data.txt'),
        os.path.join(root_dir, 'combined_safe_data.txt'),
        header=None,
        delimiter='\t',
        names=['label', 'URL']
    )
    logger.info(f"검증 정상 데이터 로드 완료: {len(benign_df)}개 샘플")

    # 데이터 결합
    df = pd.concat([phishing_df, benign_df], ignore_index=True)
    #logger.info(f"Step 1 - After Concatenation:\n{df}")

    # 결측값 제거
    df = df.dropna(subset=['URL'])
    logger.info("결측값을 제거했습니다.")
    #logger.info(f"\nStep 2 - After Dropping NaN URLs:\n{df}")

    # 레이블 매핑 (-1: 피싱 ⇒ 1, +1: 정상 ⇒ 0)
    df['label'] = df['label'].astype(int)  # 데이터 타입 변환
    label_mapping = {-1: 1, +1: 0}
    df['label'] = df['label'].map(label_mapping)
    #logger.info(f"\nStep 3 - After Label Mapping:\n{df}")

    # 데이터 셔플링
    #df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # 텍스트와 레이블 추출
    texts = df['URL'].tolist()
    labels = df['label'].tolist()

    
    # val_texts의 전체 요소 형식 확인을 위한 로그 출력
    invalid_entries = [text for text in texts if not isinstance(text, str)]
    if invalid_entries:
        logger.warning(f"val_texts에 문자열이 아닌 요소가 있습니다. 총 {len(invalid_entries)}개: {invalid_entries[:5]}")
        # 결측값 제거
        df = df.dropna(subset=['URL'])
        logger.info("결측값을 제거했습니다. 다시 확인합니다.")
        # val_texts의 전체 요소 형식 확인을 위한 로그 출력
        invalid_entries = [text for text in texts if not isinstance(text, str)]
        if invalid_entries:
            logger.warning(f"val_texts에 문자열이 아닌 요소가 있습니다. 총 {len(invalid_entries)}개: {invalid_entries[:5]}")
            # 결측값 제거
            df = df.dropna(subset=['URL'])
            logger.info("결측값을 제거했습니다.")
            # 텍스트와 레이블 추출
            texts = df['URL'].tolist()
            labels = df['label'].tolist()
        else:
            logger.info("val_texts의 모든 요소가 문자열입니다.")
        
    else:
        logger.info("val_texts의 모든 요소가 문자열입니다.")
    

    print("검증 데이터 레이블 분포:", pd.Series(labels).value_counts())

    
    #train_texts, val_texts, train_labels, val_labels
    return texts, labels


def train_model(train_texts, train_labels, val_texts, val_labels, epochs=3, batch_size=4, model=None):
    """
    모델을 훈련하고 검증 세트 성능을 출력합니다.
    """
    # 토크나이저 로드
    logger.info("토크나이저 로드 중...")
    #tokenizer = AutoTokenizer.from_pretrained("ealvaradob/bert-finetuned-phishing")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
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

    # 모델 로드 중
    if model == None:
        user_model_choice = input("새로 모델을 다운 받으시겠습니까?(y/n): ").strip().lower()
        if user_model_choice == 'y':
            logger.info("새 모델 로드 중...")
            model = AutoModelForSequenceClassification.from_pretrained("ealvaradob/bert-finetuned-phishing")
            model.to(device)
            logger.info("새 모델 로드 완료")
        elif user_model_choice == 'n':
            # 로컬 모델 로드 시도
            try:
                logger.info("로컬에 저장된 모델 로드 중...")
                model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
                model.to(device)
                logger.info("로컬 모델 로드 완료")
            except Exception as e:
                logger.error(f"로컬 모델을 로드하는 데 실패했습니다: {e}")
                print("모델을 로드할 수 없습니다. 프로그램을 종료합니다.")
                exit()
        else:
            print("올바르지 않은 입력입니다. 프로그램을 종료합니다.")
            exit()
    else:
        logger.info("기존 모델 사용 중...")

    # 옵티마이저 설정
    optimizer = AdamW(model.parameters(), lr=3e-5)

    # Gradient Accumulation 설정
    accumulation_steps = 4  # 4번의 작은 배치 후에 한 번의 업데이트

    # 모델 학습
    model.train() 
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
    accuracy = evaluate_model(model, val_loader, device)
    """
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
    """

    # 모델 저장
    logger.info("모델 저장 중...")
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(TOKENIZER_DIR)
    logger.info("모델 저장 완료")

    return model, tokenizer

def train_add_model(train_texts, train_labels, epochs=5, batch_size=4, model = None):

    # 토크나이저 로드
    logger.info("토크나이저 로드 중...")
    #tokenizer = AutoTokenizer.from_pretrained("ealvaradob/bert-finetuned-phishing")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    logger.info("토크나이저 로드 완료")

    # 데이터 토큰화
    logger.info("훈련 데이터 토큰화 중...")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors='pt')
    #val_encodings = tokenizer(val_texts, truncation=True, padding=True, return_tensors='pt')


    # PyTorch 데이터셋 생성 시 attention_mask 포함
    train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], torch.tensor(train_labels))
    #val_dataset = TensorDataset(val_encodings['input_ids'], val_encodings['attention_mask'], torch.tensor(val_labels))

    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # 모델 로드 중

    logger.info("기존 모델 사용 중...")

    # 옵티마이저 설정
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Gradient Accumulation 설정
    accumulation_steps = 4  # 4번의 작은 배치 후에 한 번의 업데이트

    # 모델 학습
    model.train() 
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

    # 모델 저장
    logger.info("모델 저장 중...")
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(TOKENIZER_DIR)
    logger.info("모델 저장 완료")

    return model, tokenizer

def evaluate_model(model, val_loader, device):
    # val_loader = DataLoader(val_dataset, batch_size=batch_size)
    """
    검증 세트에서 모델의 성능을 평가하고 정확도를 반환합니다.

    Args:
        model: PyTorch 모델 (torch.nn.Module)
        val_loader: 검증 데이터에 대한 DataLoader
        device: 모델과 데이터를 위치시킬 장치 (예: 'cuda' 또는 'cpu')

    Returns:
        검증 세트 정확도 (float)
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch[0].to(device)
            attention_mask = batch[1].to(device)
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
    return accuracy

def evaluate_and_save_results(model, val_loader, device, val_texts):
    """
    검증 세트에서 모델의 성능을 평가하고 결과를 텍스트 파일로 저장합니다.

    Args:
        model: PyTorch 모델 (torch.nn.Module)
        val_loader: 검증 데이터에 대한 DataLoader
        device: 모델과 데이터를 위치시킬 장치 (예: 'cuda' 또는 'cpu')

    Returns:
        검증 세트 정확도 (float)
    """
    model.eval()
    validation_data_size = len(val_loader.dataset)
    print("validation data size는",validation_data_size)
    correct = 0
    total = 0
    result_lines = []  # 결과를 저장할 리스트
    all_labels = []         # Initialize list to store actual labels
    all_predictions = []     # Initialize list to store predicted labels
    result_lines.append("{url} {actual_label} {predicted_label} {correctness}")

    # RESULT_DIR가 없으면 생성
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    # 고유 파일 이름 생성 (dataset_0.txt, dataset_1.txt, ...)
    file_count = len([f for f in os.listdir(RESULT_DIR) if f.startswith("dataset_") and f.endswith(".txt")])
    base_filename = os.path.join(RESULT_DIR, f"dataset_{file_count + 1}")
    result_filename = f"{base_filename}.txt"
    confusion_matrix_filename = f"{base_filename}_confusion_matrix.png"

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            inputs = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            #logger.info(f"Model is on device: {next(model.parameters()).device}")
            #logger.info(f"Inputs are on device: {inputs.device}")
            #logger.info(f"Attention mask is on device: {attention_mask.device}")
            #logger.info(f"Labels are on device: {labels.device}")

            # forward pass
            outputs = model(input_ids=inputs, attention_mask=attention_mask)
            #_, predicted = torch.max(outputs.logits, 1)
            probs = F.softmax(outputs.logits, dim=1)  # 확률 계산
            _, predicted = torch.max(probs, 1)  # 예측된 클래스 (0 또는 1)

            # Record true and predicted labels for confusion matrix
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

            for j in range(len(labels)):
                index = i * val_loader.batch_size + j  # 전체 데이터에서의 인덱스
                url = val_texts[index]  # 실제 URL 가져오기

                actual_label = "phishing" if labels[j].item() == 1 else "safe"
                predicted_label = "phishing" if predicted[j].item() == 1 else "safe"
                phishing_prob = probs[j][1].item() * 100  # 피싱 클래스(1)의 확률 (%)
                safe_prob = probs[j][0].item() * 100      # 정상 클래스(0)의 확률 (%)

                correctness = "correct" if actual_label == predicted_label else "wrong"

                #logger.info(f"URL: {val_texts[i * val_loader.batch_size + j]}")
                #logger.info(f"Actual Label: {actual_label}")
                #logger.info(f"Predicted Label: {predicted_label}")
                #logger.info(f"Probabilities -> Safe: {safe_prob:.2f}%, Phishing: {phishing_prob:.2f}%")
                
                # 결과를 텍스트 형식으로 추가
                result_lines.append(f"{url} {actual_label} {predicted_label} {correctness}")

                if correctness == "correct":
                    correct += 1
                total += 1

    # 정확도 계산
    accuracy = 100 * correct / total
    result_lines.append(f"total accuracy: {accuracy:.2f}%")
    logger.info(f"{result_filename} 정확도: {accuracy:.2f}%")

    # 결과를 파일로 저장
    with open(result_filename, "w") as f:
        f.write("\n".join(result_lines))

    # Calculate and display confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Safe", "Phishing"], yticklabels=["Safe", "Phishing"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    
    # Save confusion matrix as an image with matching filename
    plt.savefig(confusion_matrix_filename)
    logger.info(f"Confusion matrix saved to {confusion_matrix_filename}")

    return accuracy


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
        user_choice = input("작업을 선택하세요 ('start', 'run', 'train','train-more','evaluate', 'original','train-more','exit'): ").strip().lower()

        if user_choice == 'start': # 학습된 모델이 없는 상태
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

        elif user_choice == 'train': # 기존의 모델을 더 학습시키는 상태
            # 저장된 모델과 토크나이저 로드
            model, tokenizer = load_model_and_tokenizer()
            model.to(device)


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
                #predict_url(model, tokenizer)
        elif user_choice == 'evaluate':
            # 저장된 모델과 토크나이저 로드
            model, tokenizer = load_model_and_tokenizer()
            model.to(device)
            """
            if model is None or tokenizer is None:
                print("저장된 모델이 없어서 새로운 모델을 훈련합니다.")
                # 데이터 로드 및 준비
                train_texts, val_texts, train_labels, val_labels = load_and_prepare_data(ROOT_DIR)
                # 모델 훈련 및 평가
                model, tokenizer = train_model(train_texts, train_labels, val_texts, val_labels)
            """
            # 데이터 로드 및 준비
            val_texts, val_labels = load_validation_data(ROOT_DIR)  # 반환 타입 list            
            
            # DataLoader 생성
            # 토크나이저 로드
            print("val_encodings 생성 시작")
            try:
                val_encodings = tokenizer(val_texts, truncation=True, padding=True, return_tensors='pt')
            except Exception as e:
                print(f"에러 발생: {e}")
                # 개별 텍스트를 처리하는 방안
                val_encodings = [tokenizer(text, truncation=True, padding=True, return_tensors='pt') for text in val_texts]
                # 텐서로 합치기
                val_encodings = {key: torch.cat([enc[key] for enc in val_encodings], dim=0) for key in val_encodings[0]}
            #val_encodings = tokenizer(val_texts, truncation=True, padding=True, return_tensors='pt')
            print("val_encodings 완료")
            val_dataset = TensorDataset(val_encodings['input_ids'], val_encodings['attention_mask'], torch.tensor(val_labels))
            print("val_dataset 완료")
            val_loader = DataLoader(val_dataset, batch_size=4)
            print("val_loader 완료")

            accuracy = evaluate_and_save_results(model, val_loader, device, val_texts)
            print(accuracy)
            
        elif user_choice == 'original':
            # 기존 모델과 토크나이저 로드
            tokenizer = AutoTokenizer.from_pretrained("ealvaradob/bert-finetuned-phishing")
            model = AutoModelForSequenceClassification.from_pretrained("ealvaradob/bert-finetuned-phishing")
            model.to(device)
            
            # 데이터 로드 및 준비
            val_texts, val_labels = load_validation_data(ROOT_DIR)  # 반환 타입 list            
            
            # DataLoader 생성
            # 토크나이저 로드
            print("val_encodings 생성 시작")
            try:
                val_encodings = tokenizer(val_texts, truncation=True, padding=True, return_tensors='pt')
            except Exception as e:
                print(f"에러 발생: {e}")
                # 개별 텍스트를 처리하는 방안
                val_encodings = [tokenizer(text, truncation=True, padding=True, return_tensors='pt') for text in val_texts]
                # 텐서로 합치기
                val_encodings = {key: torch.cat([enc[key] for enc in val_encodings], dim=0) for key in val_encodings[0]}
            #val_encodings = tokenizer(val_texts, truncation=True, padding=True, return_tensors='pt')
            print("val_encodings 완료")
            val_dataset = TensorDataset(val_encodings['input_ids'], val_encodings['attention_mask'], torch.tensor(val_labels))
            print("val_dataset 완료")
            val_loader = DataLoader(val_dataset, batch_size=4)
            print("val_loader 완료")

            accuracy = evaluate_and_save_results(model, val_loader, device, val_texts)
            print(accuracy)

        elif user_choice == 'train-more': # 기존의 모델을 더 학습시키는 상태
            # 저장된 모델과 토크나이저 로드
            torch.cuda.empty_cache()
            model, tokenizer = load_model_and_tokenizer()
            model.to(device)
            # 데이터 로드 및 준비
            train_texts, train_labels= load_and_add_data(ROOT_DIR)
            # 기존 모델로 추가 훈련
            model, tokenizer = train_add_model(train_texts, train_labels, epochs=5, batch_size=4,model=model)
            #train_texts, train_labels, epochs=5, batch_size=4,
        
        elif user_choice == 'exit':
            print("프로그램을 종료합니다.")
            break


        else:
            print("올바르지 않은 입력입니다. 다시 시도해주세요.")
    



if __name__ == "__main__":
    main()
    
