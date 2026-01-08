from fastapi import FastAPI
from pydantic import BaseModel
from .agent import app_graph

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    # LangGraphを実行
    inputs = {"question": req.message, "logs": []}
    result = app_graph.invoke(inputs)
    
    return {
        "final_answer": result["final_answer"],
        "logs": result["logs"], # ここに思考プロセスが入っている
        "critique": result.get("critique", "") # HRの指摘内容も見せたい場合に使える
    }

# 動作確認用
@app.get("/")
def read_root():
    return {"status": "DeNA Agent is ready!"}