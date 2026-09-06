# シンプル RAG チャット (ChromaDB + Streamlit)

> **本リポジトリは main リポジトリです。**
> `main` に直接コミット・push せず、変更や試作は必ずブランチを切って
> Pull Request 経由で検証・レビューしてから取り込むこと。

ローカルの Ollama をバックエンドに、アップロードした文書を ChromaDB に
インデックスして質問応答する最小構成の RAG アプリ。

## セットアップ

```bash
pip install -r requirements.txt

# 必要なモデルを取得
ollama pull nomic-embed-text
ollama pull llama3.1:8b
```

`app.py` の `OLLAMA_URL` は `http://localhost:12000` になっている。
Ollama を既定ポートで動かしている場合は `http://localhost:11434` に変更する。

## 起動

```bash
streamlit run app.py
```

## 使い方

1. サイドバーから `.docx` / `.txt` / `.md` をアップロード
2. 「インデックス作成」を押す（チャンク化 → 埋め込み → ChromaDB へ保存）
3. チャット欄で質問する。関連チャンクを検索してプロンプトに差し込んで回答する

インデックスは `./chroma_db` に永続化されるので、再起動しても残る。

## 構成

| 項目 | 内容 |
| --- | --- |
| ベクトルDB | ChromaDB (PersistentClient) |
| 埋め込み | Ollama `nomic-embed-text` |
| 生成 | Ollama（OpenAI 互換 API 経由） |
| チャンク | 200文字 / オーバーラップ50文字 |
