import os
from langchain_community.retrievers import AzureAISearchRetriever

code_tuss_retriver = AzureAISearchRetriever(
    content_key="content",
    top_k=5,
    index_name="agente-assistente-virtual-sadt-2-index",
    api_key=os.getenv("AZURE_AI_SEARCH_API_KEY", ""),
    service_name="seu-servico-search",
)

docs = code_tuss_retriver.invoke("Dermatoscopia")

print(docs)
