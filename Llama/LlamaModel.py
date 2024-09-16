from llama_cpp import Llama
import os
import torch

# 로컬 GGUF 모델 파일 경로
model_path = "./Llama-3.1-8B-Instruct-Phishing-Classification.i1-Q4_K_M.gguf"

# 경로 존재 여부 확인을 위한 로그 추가
print(f"모델 파일 경로: {model_path}")
if not os.path.exists(model_path):
    print("오류: 모델 파일 경로가 존재하지 않습니다.")
    print("현재 작업 디렉토리:", os.getcwd())
    print("디렉토리 내용:", os.listdir(os.path.dirname(model_path) or "."))
    exit(1)

try:
    # 모델 로드
    print("모델 로드 시도 중...")
    llm = Llama(model_path=model_path, n_ctx=8192, n_threads=8, low_vmem=True)
    print("모델 로드 성공.")
except ValueError as e:
    print(f"모델 로드 실패: {e}")
    exit(1)

while True:
    # URL 입력 받기
    input_text = input("확인할 URL을 입력하세요 (종료하려면 /bye 입력): ")

    # 종료 명령 확인
    if input_text.strip().lower() == "/bye":
        print("프로그램을 종료합니다.")
        break

    # 시스템 메시지와 사용자 입력을 포맷팅
    prompt = f"<s>System: You are a helpful assistant that detects phishing websites and calculates the probability of a URL being a phishing site.</s>\n<s>Human: The URL is {input_text}. What is the probability that this is a phishing site? Provide your reasoning.</s>\n<s>Assistant:"

    try:
        # 모델 추론 수행
        print("모델 추론 수행 중...")
        output = llm(prompt, max_tokens=512, stop=["</s>"])
        generated_text = output['choices'][0]['text']

        # 결과 출력
        print("Generated Text:", generated_text)

        # 추가로, 로그 출력값 및 확률 계산
        logits = output['choices'][0].get('logits', None)
        if logits:
            # 피싱 확률 계산 (임의의 가중치로 설정)
            probabilities = torch.nn.functional.softmax(torch.tensor(logits), dim=0)
            phishing_probability = probabilities[0].item() * 100  # 첫 번째 항목의 확률을 퍼센티지로 변환

            print(f"피싱 사이트일 확률: {phishing_probability:.2f}%")
            print("판단:", generated_text)
        else:
            print("추론 결과에서 확률을 계산할 수 없습니다.")

    except Exception as e:
        print(f"모델 추론 중 오류 발생: {e}")
