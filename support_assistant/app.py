from fastapi import FastAPI
from pydantic import BaseModel, Field

from .graph import AnswerResponse, assistant_graph

app = FastAPI(title="Zepto Policy Support Assistant", version="1.0.0")


class AskRequest(BaseModel):
    query: str = Field(min_length=1)


@app.post("/ask", response_model=AnswerResponse)
def ask(request: AskRequest) -> AnswerResponse:
    result = assistant_graph.invoke({"query": request.query})
    return result["response"]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


