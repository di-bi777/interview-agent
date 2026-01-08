from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from .rag import resume_retriever, culture_retriever

load_dotenv()

# --- 1. 定義 ---
# LLMの準備
llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

# ステート（会話の状態）の定義
class AgentState(TypedDict):
    question: str       # ユーザーの質問
    context_data: str   # 検索した自分の経歴
    draft: str          # 最初の回答案
    critique: str       # DeNA人事からのダメ出し
    final_answer: str   # 最終回答
    logs: List[str]     # フロントに表示する思考ログ

# --- 2. ノード（処理担当者）の定義 ---

# Agent A: 自分専門家（事実を集めて回答案を作る）
def candidate_node(state: AgentState):
    question = state["question"]
    # 自分の経歴を検索
    docs = resume_retriever.invoke(question)
    context_text = "\n".join([d.page_content for d in docs])
    
    prompt = ChatPromptTemplate.from_template(
        "あなたは就職活動中のエンジニア学生です。以下の事実情報に基づいて、質問への回答案を作成してください。\n"
        "事実: {context}\n質問: {question}\n回答案:"
    )
    chain = prompt | llm
    response = chain.invoke({"context": context_text, "question": question})
    
    log = f"🤖 Candidate Agent: 質問「{question}」に関連する経験を検索し、ドラフトを作成しました。"
    return {"context_data": context_text, "draft": response.content, "logs": [log]}

# Agent B: DeNAカルチャー担当（ダメ出しをする）
def culture_node(state: AgentState):
    draft = state["draft"]
    # DeNAの文化を検索（全件取得に近い形でDeNAらしさを注入）
    docs = culture_retriever.invoke("DeNA Promise Delight")
    culture_text = "\n".join([d.page_content for d in docs])
    
    prompt = ChatPromptTemplate.from_template(
        "あなたは株式会社DeNAの採用担当マネージャーです。以下の回答案を厳しくチェックしてください。\n"
        "基準: {culture}\n"
        "特に「Delight（驚き）」「コトに向かう（成果思考）」の観点が足りているか確認し、"
        "足りない場合は具体的にどのように修正すべきか、簡潔に指摘してください。\n"
        "回答案: {draft}\n指摘コメント:"
    )
    chain = prompt | llm
    response = chain.invoke({"culture": culture_text, "draft": draft})
    
    log = f"🏢 DeNA HR Agent: 回答案をレビュー中... DeNAの価値観（Delight等）と照らし合わせ、改善点を指摘します。"
    return {"critique": response.content, "logs": [log]}

# Agent C: 最終調整担当（書き直す）
def writer_node(state: AgentState):
    draft = state["draft"]
    critique = state["critique"]
    
    prompt = ChatPromptTemplate.from_template(
        "指摘事項を踏まえて、回答案を最高のものに書き直してください。\n"
        "元の案: {draft}\n指摘: {critique}\n"
        "修正後の回答（丁寧かつ熱意を持って）:"
    )
    chain = prompt | llm
    response = chain.invoke({"draft": draft, "critique": critique})
    
    log = f"✍️ Writer Agent: 指摘を受け、DeNAカルチャーにフィットするように回答をブラッシュアップしました。"
    return {"final_answer": response.content, "logs": [log]}

# --- 3. グラフの構築 ---
workflow = StateGraph(AgentState)

workflow.add_node("candidate", candidate_node)
workflow.add_node("hr_review", culture_node)
workflow.add_node("writer", writer_node)

workflow.set_entry_point("candidate")
workflow.add_edge("candidate", "hr_review")
workflow.add_edge("hr_review", "writer")
workflow.add_edge("writer", END)

# コンパイル
app_graph = workflow.compile()