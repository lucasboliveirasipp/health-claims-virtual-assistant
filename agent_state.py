from typing import Annotated, TypedDict, List, Dict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

#STATE
class AgentState(TypedDict):
    """
    Messages: para armazenar histórico de interações
    Query: Pergunta do usuário
    Index: para identificar base de conhecimento aplicada
    Docs: lista de documentos recuperados e filtrados
    """

    messages: Annotated[List[BaseMessage], add_messages]
    query: str
    index: str
    docs: List[Dict]
    summary: str


class GradeDocuments(BaseModel):
    """
    Score binário para avaliar a relevância do texto para a query apresentada.
    Os valores de resposta podem ser 'yes' ou 'no'.
    Esses documentos devem fundamentar uma resposta ao assunto que foi perguntado.
    """

    binary_score: str = Field(
        description="Resposta se o produto recuperado é relevante para a necessidade do cliente, 'yes' or 'no'"
    )
