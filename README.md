# TMI 챗봇 프로젝트

## 개요
본 프로젝트는 카카오톡 대화 로그와 TMI(TMI.csv) 데이터를 기반으로 한 RAG(Retrieval-Augmented Generation) 챗봇입니다.  
LangChain과 OpenAI API를 활용해 벡터 검색 및 GPT 기반 답변 생성을 구현했으며, Streamlit UI를 통해 손쉽게 질문하고 대화를 기록할 수 있습니다.

---

## 주요 기능
- **대화 로그 및 TMI 데이터 파싱:** 카카오톡 대화 텍스트와 CSV TMI 데이터를 읽어와 문서화
- **벡터 임베딩 및 검색:** OpenAI 임베딩으로 문서 임베딩 후 FAISS 벡터 스토어 구축
- **RAG 질의응답:** 사용자의 질문에 관련 문서를 검색하여 GPT가 답변 생성
- **대화 기록 저장:** SQLite 데이터베이스에 질의응답 내역 저장 및 불러오기
- **Streamlit 기반 인터페이스:** 웹에서 질문 입력 및 대화 내역 확인 가능
- **강아지 이름 추출 기능:** LLM 기반으로 대화 속 강아지 이름 후보를 자동 추출
- **워드클라우드 시각화:** 대화 속 등장 빈도가 높은 강아지 이름을 시각화

---

## 설치 및 실행

### 필수 조건
- Python 3.8 이상
- OpenAI API Key 준비 (환경 변수 `OPENAI_API_KEY`에 설정)

### 주요 라이브러리 설치
```bash
pip install -r requirements.txt

