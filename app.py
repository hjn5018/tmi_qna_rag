import streamlit as st
import sqlite3
import re
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
from datetime import datetime
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

config = {
    "kakao_path": "data/kakao_chat.txt",
    "csv_path": "data/tmi.csv",
    "db_path": "db/chat_history.db",
    "vector_path": "db/tmi_faiss",
    "chunk_size": 500,
    "chunk_overlap": 20,
    "llm_model": "gpt-3.5-turbo",
    "chat_history_limit": 10,
    "font_path": "malgun.ttf",  # mac 사용자는 "NanumGothic.ttf" 등으로 교체
}

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

def load_kakao_text(filepath):
    with open(filepath, encoding="utf-8") as f:
        return f.read()

def load_tmi_csv(filepath):
    df = pd.read_csv(filepath)
    return [Document(page_content=row["tmi"]) for _, row in df.iterrows()]

def load_all_documents():
    kakao_docs = load_kakao_chat(config["kakao_path"])
    tmi_docs = load_tmi_csv(config["csv_path"])
    return kakao_docs + tmi_docs

def build_vectorstore(documents):
    splitter = CharacterTextSplitter(chunk_size=config["chunk_size"], chunk_overlap=config["chunk_overlap"])
    split_docs = splitter.split_documents(documents)

    embedding = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(split_docs, embedding)
    vectorstore.save_local(config["vector_path"])
    return vectorstore

def save_chat(question, answer):
    conn = sqlite3.connect(config["db_path"])
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

def load_chat_history(limit=None):
    conn = sqlite3.connect(config["db_path"])
    cursor = conn.cursor()
    limit_clause = f"LIMIT {limit}" if limit else ""
    cursor.execute(f"SELECT question, answer FROM chat_log ORDER BY id DESC {limit_clause}")
    rows = cursor.fetchall()
    conn.close()
    return rows[::-1]

def answer_query(query, vectorstore):
    retriever = vectorstore.as_retriever()
    related_docs = retriever.invoke(query)
    tmi_context = "\n".join([doc.page_content for doc in related_docs]) or "(관련 정보 없음)"

    include_history = any(k in query for k in ["이전", "전에", "기억", "했었", "했니", "대화", "과거", "무슨 질문"])
    history_text = ""
    if include_history:
        history = load_chat_history(limit=config["chat_history_limit"])
        history_text = "\n".join([f"User: {q}\nAssistant: {a}" for q, a in history])

    messages = [
        SystemMessage(content="당신은 사용자에게 친절하고 구체적인 어시스턴트입니다."),
        HumanMessage(content=f"""# 📘 TMI 및 대화 데이터:
{tmi_context}

{f"# 📜 이전 대화:\n{history_text}" if history_text else ""}
# 🤔 질문:
{query}""")
    ]

    llm = ChatOpenAI(model=config["llm_model"])
    response = llm.invoke(messages)
    return response.content.strip()

# === 강아지 이름 분석 ===
def extract_dog_names_with_llm(kakao_text: str) -> list:
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    example = (
        "[ㅎㅎ] [오후 8:44] 푸들 한 아이는 집에 가고 작고 까불까불하는 푸들과 일요일까지 같이 있습니다~\n"
        "[ㅎㅎ] [오후 9:04] 몇 번 만난 적 있는 소심한 포메 여아가 왔습니다.\n"
        "[ㅎㅎ] [오전 11:31] 장고랑 산책하고 왔어요\n"
        "[ㅎㅎ] [오전 11:32] 루이도 같이 있었어요\n"
    )
    system_prompt = (
        "다음은 강아지에 대한 카카오톡 대화입니다.\n"
        "- 강아지는 주로 산책하거나 사진에 찍히거나 간식을 받습니다.\n"
        "- 사람 이름, 날짜, 장소, 시간 표현은 제외하세요.\n"
        "- 강아지 이름만 Python 리스트 형식으로 반환해 주세요. 예: ['장고', '루이', '포메', '푸들']\n"
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=example + kakao_text[:3000])
    ]

    response = llm.invoke(messages)
    try:
        names = eval(response.content.strip())
        return [n for n in names if isinstance(n, str) and re.fullmatch(r"[가-힣]{2,4}", n)]
    except:
        return []

def count_name_occurrences(text, name_list):
    words = re.findall(r"[가-힣]{2,}", text)
    return Counter([w for w in words if w in name_list])

def plot_wordcloud(counter):
    if not counter:
        st.warning("감지된 이름이 없어요.")
        return
    wc = WordCloud(
        font_path=config["font_path"],
        background_color="white",
        width=800,
        height=400
    ).generate_from_frequencies(counter)

    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)

# === Streamlit 앱 실행 ===
def run_app():
    st.set_page_config(page_title="TMI 챗봇", layout="centered")
    tab1, tab2 = st.tabs(["💬 TMI 챗봇", "🐶 강아지 이름 분석"])

    # === TAB 1: 챗봇 ===
    with tab1:
        st.header("💬 TMI 챗봇")

        if "vectorstore" not in st.session_state:
            docs = load_all_documents()
            st.session_state.vectorstore = build_vectorstore(docs)

        if "history" not in st.session_state:
            st.session_state.history = []

        query = st.text_input("질문을 입력하세요:", "")

        if st.button("질문하기") and query.strip():
            with st.spinner("답변 생성 중..."):
                answer = answer_query(query, st.session_state.vectorstore)
                save_chat(query, answer)
                st.session_state.history.append((query, answer))

        if st.session_state.history:
            st.write("## 대화 기록")
            for i, (q, a) in enumerate(st.session_state.history):
                st.markdown(f"**Q{i+1}:** {q}")
                st.markdown(f"**A{i+1}:** {a}")

    # === TAB 2: 강아지 이름 워드클라우드 ===
    with tab2:
        st.header("🐶 카카오톡 속 강아지 이름 워드클라우드@@@@")
        if st.button("강아지 이름 분석 시작"):
            with st.spinner("LLM이 강아지 이름을 추출 중입니다..."):
                kakao_text = load_kakao_text(config["kakao_path"])
                dog_names = extract_dog_names_with_llm(kakao_text)

                if dog_names:
                    st.success(f"강아지 이름: {', '.join(dog_names)}")
                    counts = count_name_occurrences(kakao_text, dog_names)
                    st.write("### 📊 이름별 등장 횟수")
                    st.json(dict(counts))
                    st.write("### ☁️ 워드클라우드")
                    plot_wordcloud(counts)
                else:
                    st.warning("강아지 이름을 찾을 수 없었어요.")

if __name__ == "__main__":
    run_app()
