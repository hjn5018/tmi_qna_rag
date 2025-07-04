import streamlit as st
import sqlite3
import re
import pandas as pd
from datetime import datetime

from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage

from dotenv import load_dotenv

load_dotenv()

# 경로
KAKAO_PATH = "data/kakao_chat.txt"
CSV_PATH = "data/tmi.csv"
DB_PATH = "db/chat_history.db"
VECTOR_PATH = "db/tmi_faiss"

def load_kakao_chat(filepath):
    with open(filepath, encoding="utf-8") as f:
        lines = f.readlines()

    documents = []
    current_date = ""
    msg_pattern = re.compile(r"\[(.*?)\] \[(.*?)\] (.+)")

    for line in lines:
        line = line.strip()
        if "---------------" in line:
            date_match = re.search(r"\d{4}년 \d{1,2}월 \d{1,2}일", line)
            if date_match:
                current_date = date_match.group(0)
            continue

        match = msg_pattern.match(line)
        if match:
            sender, time, content = match.groups()
            full_msg = f"날짜: {current_date}, 시간: {time}, 보낸이: {sender}\n내용: {content}"
            documents.append(Document(page_content=full_msg))

    return documents

def load_tmi_csv(filepath):
    df = pd.read_csv(filepath)
    return [Document(page_content=row["tmi"]) for _, row in df.iterrows()]

def ingest_all():
    kakao_docs = load_kakao_chat(KAKAO_PATH)
    tmi_docs = load_tmi_csv(CSV_PATH)
    docs = kakao_docs + tmi_docs

    splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    split_docs = splitter.split_documents(docs)

    embedding = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(split_docs, embedding)
    vectorstore.save_local(VECTOR_PATH)

def save_chat(question, answer):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT,
            answer TEXT,
            timestamp TEXT
        )
    """)
    cursor.execute(
        "INSERT INTO chat_log (question, answer, timestamp) VALUES (?, ?, ?)",
        (question, answer, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()

def load_chat_history(limit=10):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT question, answer FROM chat_log ORDER BY id DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    return rows[::-1]

def answer_query(query):
    ingest_all()

    embedding = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(VECTOR_PATH, embedding, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever()
    related_docs = retriever.invoke(query)
    tmi_context = "\n".join([doc.page_content for doc in related_docs]) or "(관련 정보 없음)"

    include_history = any(k in query for k in ["이전", "전에", "기억", "했었", "했니", "대화", "과거", "무슨 질문"])

    if include_history:
        history = load_chat_history()
        history_text = "\n".join([f"User: {q}\nAssistant: {a}" for q, a in history])
    else:
        history_text = ""

    messages = [
        SystemMessage(content="당신은 사용자에게 친절하고 구체적인 어시스턴트입니다."),
        HumanMessage(content=f"""# 📘 TMI 및 대화 데이터:
{tmi_context}

{f"# 📜 이전 대화:\n{history_text}" if history_text else ""}
# 🤔 질문:
{query}""")
    ]

    llm = ChatOpenAI(model="gpt-3.5-turbo")
    response = llm.invoke(messages)
    return response.content.strip()

# Streamlit 앱
st.title("TMI 챗봇 (Streamlit)")

if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_input("질문을 입력하세요:", "")

if st.button("질문하기") and query.strip() != "":
    with st.spinner("답변 생성 중..."):
        answer = answer_query(query)
        save_chat(query, answer)
        st.session_state.history.append((query, answer))

if st.session_state.history:
    st.write("## 대화 기록")
    for i, (q, a) in enumerate(st.session_state.history):
        st.markdown(f"**Q{i+1}:** {q}")
        st.markdown(f"**A{i+1}:** {a}")
