from llama_cpp import Llama
import torch

# 로컬 GGUF 모델 파일 경로
model_path = "./Llama-3.1-8B-Instruct-Phishing-Classification.i1-Q4_K_M.gguf"

# 모델 로드
llm = Llama(model_path=model_path, n_ctx=4096, n_threads=8)

# URL 입력 받기
input_text = input("확인할 URL을 입력하세요: ")

# 시스템 메시지와 사용자 입력을 포맷팅
prompt = f"<s>System: You are a helpful assistant that detects phishing websites.</s>\n<s>Human: {input_text}</s>\n<s>Assistant:"

# 모델 추론 수행
output = llm(prompt, max_tokens=128, stop=["</s>"])

# 출력된 텍스트 추출
generated_text = output['choices'][0]['text']

# 로그 출력값 및 확률 계산
# `llama-cpp-python`에서 logits을 직접적으로 구할 수는 없으나, 아래는 출력된 텍스트의 기본 출력
print("Generated Text:", generated_text)

# 만약 확률 계산이 가능하다면, 여기에 확률 변환 코드 추가
# 이 예시에서는 softmax를 사용해 직접적으로 logits을 계산하는 부분은 생략했습니다.
# 모델이 텍스트 생성 기반일 때 로그 확률을 제공하는지 확인 필요

# 로그 출력값과 확률을 유사하게 출력하도록 작성
print("Logits: 로그 출력값을 직접적으로 제공하지 않음 (모델 출력 기반)")
print("Probabilities: 확률 변환을 위한 추가 계산이 필요")
