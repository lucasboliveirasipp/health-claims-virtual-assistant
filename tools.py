# Carrega variáveis do arquivo .env mais próximo (pasta do projeto)
def _carregar_env():
    from pathlib import Path
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    for pasta in Path(__file__).resolve().parents:
        if (pasta / ".env").exists():
            load_dotenv(pasta / ".env")
            break


_carregar_env()

from langchain_core.tools import tool
import os
from langchain_community.retrievers import (
    AzureCognitiveSearchRetriever,
    AzureAISearchRetriever,
)

#####
# A ideia do agente é consultar a tabela de excel
# Temos novos materiais - fazer a indexação
# Atualizar o index: tuss-saude-index
# Antes de atualizar os indexes, refatorar a aplicação para que ela responda tanto pelo agente rag (usando langchain) (implentar) quanto pelo agente que ja existe (excel)
# O agente rag vai fazer a busca no index 'reembolso_saude_index'
#####


os.environ["API_VERSION"] = "2024-02-15-preview"

os.environ["AZURE_COGNITIVE_SEARCH_SERVICE_NAME"] = "seu-servico-search"
os.environ["AZURE_AI_SEARCH_SERVICE_NAME"] = "seu-servico-search"


code_tuss_retriver = AzureAISearchRetriever(
    content_key="content",
    top_k=5,
    index_name="agente-assistente-virtual-sadt-index",
    api_key=os.getenv("AZURE_AI_SEARCH_API_KEY", ""),
    service_name="seu-servico-search",
)

code_tuss_retriver_2 = AzureAISearchRetriever(
    content_key="content",
    top_k=4,
    index_name="agente-assistente-virtual-sadt-2-index",
    api_key=os.getenv("AZURE_AI_SEARCH_API_KEY", ""),
    service_name="seu-servico-search",
)

code_tuss_retriver_3 = AzureAISearchRetriever(
    content_key="content",
    top_k=4,
    index_name="agente-assistente-virtual-sadt-3-index",
    api_key=os.getenv("AZURE_AI_SEARCH_API_KEY", ""),
    service_name="seu-servico-search",
)

code_tuss_retriver_5 = AzureAISearchRetriever(
    content_key="content",
    top_k=2,
    index_name="agente-assistente-virtual-sadt-5-index",
    api_key=os.getenv("AZURE_AI_SEARCH_API_KEY", ""),
    service_name="seu-servico-search",
)

# HP_retriever = AzureAISearchRetriever(
#     content_key="content", top_k=5, index_name="agente-assistente-virtual-hp-index", api_key=os.getenv("AZURE_AI_SEARCH_API_KEY", ""), service_name="seu-servico-search",
# )

# HM_retriever = AzureAISearchRetriever(
#     content_key="content", top_k=5, index_name="agente-assistente-virtual-hm-index", api_key=os.getenv("AZURE_AI_SEARCH_API_KEY", ""), service_name="seu-servico-search",
# )"SADT_Multi_Index"


@tool("SADT_retriever_tool")
def SADT_retriever_tool(query: str):
    """
    Ferramenta de consulta nas informações do campo SADT da acme seguros
    Args:
        query (str): Query elaborada a partir da pergunta do usuário.

    Returns:
        dict: Um dicionário contendo a query original, os documentos relevantes encontrados (docs) e o índice consultado ("agente-assistente-virtual-sadt-index").
    """

    docs_3 = code_tuss_retriver_3.invoke(query)

    docs_2 = code_tuss_retriver_2.invoke(query)

    docs = code_tuss_retriver.invoke(query)

    docs_5 = code_tuss_retriver.invoke(query)

    combined_docs = docs + docs_2 + docs_3

    return {
        "query": query,
        "docs": combined_docs,
        "index": [
            "agente-assistente-virtual-sadt-3-index",
            "agente-assistente-virtual-sadt-2-index",
            "agente-assistente-virtual-sadt-index",
        ],
    }


# @tool('HP_retriever_tool')
# def HP_retriever_tool(query: str):
#     """
#     Ferramenta de consulta nas informações do Códigos TUSS da Acme Seguros.
#     Exemplos da estruturas de querys que podem ser feitas na ferremanta:
#     - Código tuss para septoplastia
#     - Código tuss para exame de sangue

#     Args:
#         query (str): Query elaborada a partir da pergunta do usuário relacionada a procedimentos e Códigos TUSS.

#     Returns:
#         dict: Um dicionário contendo a query original, os documentos relevantes encontrados (docs) e o índice consultado ("tuss-saude-index").
#     """
#     docs =  HP_retriever.invoke(query)
#     return {"query": query, "docs": docs, "index": "tuss-saude-index"}

# @tool('HM_retriever_tool')
# def HM_retriever_tool(query: str):
#     """
#     Ferramenta de consulta nas informações de Reembolso de Saúde nos documentos interno.
#     Realiza buscas apenas envolvendo os seguintes tópicos: reembolso, cobertura, pagamentos, reembolsos de planos, ou aspectos do Código de Defesa do Consumidor
#     Exemplos da estruturas de querys que podem ser feitas na ferremanta:
#     - Reembolso consulta médica
#     - Cobertura do plano Brasil top plus
#     - Diretrizes {nome do procedimento}

#     Args:
#         query: Query elaborada a partir da pergunta do usuário referente a procedimentos e leis.
#     Returns:
#         str: Um dicionário contendo a query original, os documentos relevantes encontrados (docs).
#     """
#     docs = HM_retriever.invoke(query)
#     return {"query": query, "docs": docs, "index": "reembolso-saude-index"}
