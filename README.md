# SeQR-AI
AI machine learning model for SeQR 

## ⏰ 개발 기간

- 24.09.15 - 진행중

## Version Requirements - myenv
tensorflow==2.12.0
keras==2.12.0
transformers==4.31.0
Python 3.10
scikit-learn
pandas
pytorch
git LFS

# 참고 - LFS 파일을 다운로드하는 방법
프로젝트를 복제할 때, LFS 파일을 자동으로 다운로드하려면 다음과 같이 Git LFS를 설치해야 합니다. 일반적으로 git clone 명령을 실행하면 LFS 파일도 함께 다운로드되지만, git lfs install 명령으로 LFS를 활성화해야만 해당 파일을 제대로 다운로드할 수 있습니다.

git clone https://github.com/TeamSeQR/SeQR-AI.git
git lfs install
git lfs pull



## 📝 규칙

#### 커밋 컨벤션
  - "태그: 한글 커밋 메시지" 형식으로 작성
  - 컨벤션 예시
    - feat: 새로운 기능 추가, 기존의 기능을 요구 사항에 맞추어 수정
    - fix: 버그 수정
    - docs: 문서 수정
    - style: 코드 포맷팅, 오타 수정, 주석 수정 및 삭제 등
    - refactor: 코드 리팩터링
    - chore: 빌드 및 패키지 수정 및 삭제
    - merge: 브랜치를 머지
    - ci: CI 관련 설정 수정
    - test: 테스트 코드 추가/수정
    - release: 버전 릴리즈