import os
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# 環境変数の読み込み
load_dotenv()

# パス設定
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_DIR = os.path.join(BASE_DIR, "chroma_db") # ここにDBを保存します

def create_vector_db():
    # 既存のDBがあれば削除（クリーンな状態で作り直すため）
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
        print(f"既存のDBを削除しました: {DB_DIR}")

    documents = []
    
    # 読み込むファイルリスト
    files = ["resume.txt", "dena_culture.txt"]

    for f in files:
        file_path = os.path.join(DATA_DIR, f)
        if os.path.exists(file_path):
            print(f"Loading {f}...")
            loader = TextLoader(file_path, encoding='utf-8')
            docs = loader.load()
            documents.extend(docs)
        else:
            print(f"Warning: {f} not found.")

    # テキスト分割
    text_splitter = CharacterTextSplitter(
        separator="\n\n", # 前回の工夫：段落で切る
        chunk_size=1000,
        chunk_overlap=50,
    )
    split_docs = text_splitter.split_documents(documents)

    # DB作成と保存 (persist_directoryを指定するのがポイント)
    print(f"Creating embeddings for {len(split_docs)} chunks...")
    Chroma.from_documents(
        documents=split_docs,
        embedding=OpenAIEmbeddings(),
        persist_directory=DB_DIR
    )
    print("✅ Database created successfully!")

if __name__ == "__main__":
    create_vector_db()