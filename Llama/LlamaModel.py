from llama_cpp import Llama
import os
import torch
import re

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

    # 시스템 메시지와 사용자 입력을 JSON 형식으로 대답하도록 제한
    prompt = f"""<s>System: You are a helpful assistant that detects phishing websites and calculates the probability of a URL being a phishing site. 
    Answer in the following JSON format: {{ "probability": 0.XX, "reason": "..." }}</s>\n<s>Human: The URL is {input_text}. What is the probability that this is a phishing site? Provide your reasoning.</s>\n<s>Assistant:"""

    try:
        # 모델 추론 수행
        print("모델 추론 수행 중...")
        output = llm(prompt, max_tokens=512, stop=["</s>"])
        generated_text = output if isinstance(output, str) else output['choices'][0]['text']
        print("Generated Text:", generated_text)

        # 정규 표현식을 사용하여 확률 값 추출
        match = re.search(r'"probability":\s*0\.(\d+)', generated_text)
        if match:
            phishing_probability = float(f"0.{match.group(1)}") * 100  # 확률을 퍼센티지로 변환
            print(f"피싱 사이트일 확률: {phishing_probability:.2f}%")
        else:
            print("피싱 사이트일 확률을 계산할 수 없습니다.")

        # 이유 출력
        match_reason = re.search(r'"reason":\s*"([^"]+)"', generated_text)
        if match_reason:
            print("판단 이유:", match_reason.group(1))
        else:
            print("판단 이유를 찾을 수 없습니다.")

    except Exception as e:
        print(f"모델 추론 중 오류 발생: {e}")
