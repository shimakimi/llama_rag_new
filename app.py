"""シンプルな RAG チャットアプリ (ChromaDB + Streamlit + Ollama)"""

import streamlit as st
from openai import OpenAI
import chromadb
from docx import Document
import requests

# ---------------- 設定 ----------------
OLLAMA_URL = "http://localhost:12000"   # Ollama のデフォルトは http://localhost:11434
EMBED_MODEL = "nomic-embed-text"
DB_DIR = "./chroma_db"
COLLECTION_NAME = "local_docs"

CHUNK_SIZE = 200
CHUNK_OVERLAP = 50


# ---------------- ChromaDB ----------------
@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path=DB_DIR)
    return client.get_or_create_collection(name=COLLECTION_NAME)


collection = get_collection()


# ---------------- ユーティリティ ----------------
def ollama_embed(text):
    """Ollama の埋め込みモデルでテキストをベクトル化する"""
    r = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["embedding"]


def load_document(file):
    """アップロードされたファイルからテキストを取り出す"""
    if file.name.lower().endswith(".docx"):
        return "\n".join(p.text for p in Document(file).paragraphs)
    return file.read().decode("utf-8", errors="ignore")


def split_text(text):
    """テキストをオーバーラップ付きで固定長に分割する"""
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start:start + CHUNK_SIZE].strip()
        if chunk:
            chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# ---------------- サイドバー ----------------
st.set_page_config(page_title="Local RAG Chat")

st.sidebar.title("設定")
model = st.sidebar.text_input("モデル名", value="llama3.1:8b")
temperature = st.sidebar.slider("temperature", 0.0, 2.0, 0.3, 0.1)
n_results = st.sidebar.slider("参照するチャンク数", 1, 10, 3)
system_prompt = st.sidebar.text_area(
    "System Prompt",
    "あなたは有能なアシスタントです。日本語で回答してください。",
)

st.sidebar.divider()
st.sidebar.subheader("ナレッジベース")
st.sidebar.caption(f"登録済みチャンク数: {collection.count()}")

uploaded_files = st.sidebar.file_uploader(
    "ファイルをアップロード (.docx / .txt / .md)",
    type=["docx", "txt", "md"],
    accept_multiple_files=True,
)

if st.sidebar.button("インデックス作成", use_container_width=True):
    if not uploaded_files:
        st.sidebar.warning("ファイルを選択してください")
    else:
        progress = st.sidebar.progress(0.0, text="インデックス作成中...")
        for f_idx, file in enumerate(uploaded_files):
            chunks = split_text(load_document(file))
            if not chunks:
                continue
            collection.upsert(
                documents=chunks,
                embeddings=[ollama_embed(c) for c in chunks],
                metadatas=[{"source": file.name, "chunk": i} for i in range(len(chunks))],
                ids=[f"{file.name}_{i}" for i in range(len(chunks))],
            )
            progress.progress((f_idx + 1) / len(uploaded_files), text=f"{file.name} 完了")
        progress.empty()
        st.sidebar.success("インデックス作成完了")
        st.rerun()

if st.sidebar.button("ナレッジベースを削除", use_container_width=True):
    ids = collection.get()["ids"]
    if ids:
        collection.delete(ids=ids)
    st.sidebar.success("削除しました")
    st.rerun()

st.sidebar.divider()
if st.sidebar.button("会話をリセット", use_container_width=True):
    st.session_state.messages = []
    st.rerun()


# ---------------- チャット ----------------
st.title("Local RAG Chat")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

client = OpenAI(api_key="ollama", base_url=f"{OLLAMA_URL}/v1")

prompt = st.chat_input("メッセージを入力")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # --- 検索 (RAG) ---
    docs, metas = [], []
    if collection.count() > 0:
        results = collection.query(
            query_embeddings=[ollama_embed(prompt)],
            n_results=min(n_results, collection.count()),
        )
        docs = results["documents"][0]
        metas = results["metadatas"][0]

    if docs:
        context_text = "\n\n".join(f"[{i + 1}] {d}" for i, d in enumerate(docs))
        user_content = (
            "以下は関連ドキュメントの抜粋です。\n"
            f"{context_text}\n\n"
            "この情報を参考に、以下の質問に答えてください。"
            "ドキュメントに答えがない場合は、その旨を伝えてください。\n"
            f"質問: {prompt}"
        )
    else:
        user_content = prompt

    # 履歴には元の質問を残し、LLM への送信時だけ検索結果を差し込む
    messages = list(st.session_state.messages[:-1])
    messages.append({"role": "user", "content": user_content})
    if system_prompt.strip():
        messages.insert(0, {"role": "system", "content": system_prompt})

    # --- 生成 ---
    with st.chat_message("assistant"):
        placeholder = st.empty()
        answer = ""
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                answer += delta
                placeholder.write(answer)

        if docs:
            with st.expander(f"参照ドキュメント ({len(docs)}件)"):
                for i, (d, m) in enumerate(zip(docs, metas), start=1):
                    st.markdown(f"**[{i}] {m.get('source', '?')}**")
                    st.caption(d)

    st.session_state.messages.append({"role": "assistant", "content": answer})
