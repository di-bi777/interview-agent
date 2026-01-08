import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
load_dotenv()  #
# DBのパス
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "chroma_db")

def get_retriever():
    # 保存済みのDBをロードする
    if not os.path.exists(DB_DIR):
        raise FileNotFoundError("Vector DBが見つかりません。'python -m backend.build_db' を実行してください。")
    
    db = Chroma(
        persist_directory=DB_DIR,
        embedding_function=OpenAIEmbeddings()
    )
    return db.as_retriever()

# エージェントがimportするときに呼び出される
resume_retriever = get_retriever()
# 同じDBから検索しますが、Metadataフィルタなどを使わない簡易版なので、
# 今回は同じRetrieverを使い回しても、DeNA知識も自分知識も全部検索対象になります。
# (精度を上げるならMetadataで区別しますが、一旦全部入りで検索させても大きな問題はありません)
culture_retriever = resume_retriever