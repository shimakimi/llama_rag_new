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

1. サイドバーから `.docx` / `.pdf` / `.txt` / `.md` をアップロード
2. 「インデックス作成」を押す（チャンク化 → 埋め込み → ChromaDB へ保存）
3. チャット欄で質問する。関連チャンクを検索してプロンプトに差し込んで回答する
4. 回答の下の「参照ドキュメント」を開くと、各チャンクの類似度をグラフ（棒グラフ）で確認できる

インデックスは `./chroma_db` に永続化されるので、再起動しても残る。

## 構成

| 項目 | 内容 |
| --- | --- |
| ベクトルDB | ChromaDB (PersistentClient, cosine距離) |
| 埋め込み | Ollama `nomic-embed-text` |
| 生成 | Ollama（OpenAI 互換 API 経由） |
| チャンク | 200文字 / オーバーラップ50文字 |
| 対応ファイル形式 | `.docx` / `.pdf` / `.txt` / `.md` |

## ブランチでの試作検証 (`feature/add-rag`)

`feature/add-rag` ブランチでは、以下の機能追加を試作・検証する。

- 画像アップロード対応（画像を取り込んでインデックス・検索に利用）
- 分析機能（アップロード文書やチャット履歴に対する分析・可視化）
- **Word (`.docx`) / PDF (`.pdf`) アップロード対応**（`pypdf` で PDF テキストを抽出し、既存の docx 対応と同じチャンク化・インデックスフローに統合）— 実装済み
- **類似度グラフ出力**（チャット回答時、ChromaDB の検索結果（cosine距離から算出した類似度）を「参照ドキュメント」内に棒グラフで表示） — 実装済み

検証が完了し次第、PR を作成して `main` にレビュー・マージする。
