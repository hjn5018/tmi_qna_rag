import re
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv

load_dotenv()

# 경로 설정
KAKAO_PATH = "data/kakao_chat.txt"
VECTOR_DB_PATH = "db/tmi_faiss"

# 카카오톡 메시지 정규식 패턴
MESSAGE_PATTERN = re.compile(r"\[(.*?)\] \[(.*?)\] (.+)")

def load_kakao_chat(filepath):
    with open(filepath, encoding="utf-8") as f:
        lines = f.readlines()

    documents = []
    current_date = ""
    for line in lines:
        line = line.strip()

        # 날짜 라인
        if "---------------" in line:
            date_match = re.search(r"\d{4}년 \d{1,2}월 \d{1,2}일", line)
            if date_match:
                current_date = date_match.group(0)
            continue

        match = MESSAGE_PATTERN.match(line)
        if match:
            sender, time, content = match.groups()
            full_message = f"{current_date} {time} | {sender}: {content}"
            documents.append(Document(page_content=full_message))

    return documents

def ingest_kakao():
    print("📥 kakao_chat.txt 읽는 중...")
    docs = load_kakao_chat(KAKAO_PATH)

    print(f"🔢 전체 메시지 수: {len(docs)}")
    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0)
    split_docs = splitter.split_documents(docs)

    embedding = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(split_docs, embedding)
    vectorstore.save_local(VECTOR_DB_PATH)
    print("✅ 벡터 저장 완료:", VECTOR_DB_PATH)

    # print("💬 예시 메시지:")
    # for d in docs[:5]:
    #     print(d.page_content)


if __name__ == "__main__":
    ingest_kakao()
    