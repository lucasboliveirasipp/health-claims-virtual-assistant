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

from langchain_openai import AzureChatOpenAI
from pymongo import MongoClient
import os

os.environ["API_VERSION"] = "2024-08-01-preview"
os.environ["GPT4"] = "gpt-4o-mini"

llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
    openai_api_version="2024-08-01-preview",
    deployment_name="gpt-4o-mini",
    openai_api_type="azure",
    temperature=0,
)
