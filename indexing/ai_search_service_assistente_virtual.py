import os
from PyPDF2 import PdfReader
import pandas as pd
import docx
from langchain_text_splitters import TokenTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_openai import AzureOpenAIEmbeddings
from openpyxl import load_workbook
import openai
from tqdm import tqdm
import time
import requests
import json
from glob import glob
from typing import List
from typing import Dict, List, Any
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from documents_utils import (
    clean_key,
    extract_pages_from_pdf_simple_and_format_tables,
)


embeddings = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-ada-002",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
    api_version="2023-05-15",
    api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
)


def create_index_v2(index_name: str, semantic_name: str, columns: list):
    """
    Cria um índice no serviço Azure Cognitive Search.

    Args:
        index_name (str): O nome do índice a ser criado.
        semantic_name (str): O nome semântico do índice.
        columns (list): Uma lista de dicionários contendo as informações das colunas a serem criadas.

    Returns:
        dict: Um dicionário contendo o template da requisição de criação do índice.

    """
    relative_path = os.path.join("template_index.json")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, relative_path)
    try:
        with open(file_path, "r") as file:
            createIndexTemplate = json.load(file)
    except FileNotFoundError:
        print(f"❌ Erro: Arquivo de template não encontrado em: {file_path}")
        return
    createIndexTemplate["name"] = index_name
    createIndexTemplate["semantic"]["configurations"][0]["name"] = semantic_name
    # Colunas padrão
    default_fields = [
        {
            "name": "id",
            "type": "Edm.String",
            "key": True,
            "searchable": False,
            "filterable": False,
        },
        {
            "name": "content",
            "type": "Edm.String",
            "searchable": True,
            "retrievable": True,
            "analyzer": "pt-Br.lucene",
        },
        {
            "name": "contentVector",
            "type": "Collection(Edm.Single)",
            "searchable": True,
            "retrievable": True,
            "dimensions": 1536,
            "vectorSearchProfile": "vector-profile-1700855443968",
        },
    ]

    # Adiciona colunas de metadados
    for col in columns:
        # Garante que 'id' e 'content' não sejam adicionados duas vezes
        if col.get("name") not in ["id", "content", "contentVector"]:
            default_fields.append(col)

    createIndexTemplate["fields"] = default_fields
    headers = {
        "Content-Type": "application/json",
        "api-key": os.getenv("AZURE_SEARCH_KEY", ""),  # ⚠️ PREENCHA SUA CHAVE AZURE SEARCH AQUI
    }
    # Use o endpoint do Azure Search (ajuste se necessário)
    azure_search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT", "")
    url = f"{azure_search_endpoint}/indexes/{index_name}?api-version=2023-10-01-Preview&allowIndexDowntime=true"
    try:
        response = requests.put(url, headers=headers, json=createIndexTemplate)
        if response.status_code in [200, 201, 204]:
            print(
                f"✅ Índice {index_name} criado/atualizado com sucesso. Status: {response.status_code}"
            )
        else:
            print(
                f"❌ Erro ao criar o índice {index_name}. Status: {response.status_code}. Resposta: {response.text}"
            )
    except Exception as e:
        print(f"❌ Erro de requisição ao criar o índice: {e}")


def uploadToIndex(index: str, template: Dict[str, Any]):
    """Faz o upload de documentos para um índice usando a API REST."""
    headers = {
        "Content-Type": "application/json",
        "api-key": os.getenv("AZURE_OPENAI_API_KEY", ""),
    }

    print(f"Indexando documentos em {index}...")
    try:
        response = requests.post(
            url=f"{os.getenv('AZURE_SEARCH_ENDPOINT', '').rstrip('/')}/indexes/{index}/docs/index?api-version=2023-10-01-Preview",
            headers=headers,
            json=template,
        )
        print(response.json())
    except Exception as e:
        print(f"Erro no uploadToIndex: {e}")


def embed_single_chunk(text: str) -> List[float]:
    """Retorna a incorporação (embedding) de um texto usando o objeto global AzureOpenAIEmbeddings."""
    # Usa o objeto global 'embeddings'
    return embeddings.embed_query(text)


def vectorize_and_upload_v2(
    chunks: List[Dict[str, Any]], index_name: str
) -> Dict[str, Any]:
    """
    Vetoriza e envia documentos para o VectorDB para indexação usando a API REST em lotes.
    Args:
        chunks (list): Lista de dicionários (chunks) com 'content' (conteúdo formatado).
        index_name (str): Nome do índice.
    """
    template = {"value": []}
    batch_count = 0
    index_full_name = f"{index_name}-index"

    print("Vetorizando documentos...")
    if not isinstance(chunks, list) or not all(
        isinstance(item, dict) for item in chunks
    ):
        raise ValueError("O argumento 'chunks' deve ser uma lista de dicionários.")

    for i, chunk in tqdm(enumerate(chunks)):
        # GERAÇÃO DO VETOR
        vector = embed_single_chunk(chunk.get("content", ""))
        # Cria o documento no formato da API REST (Azure AI Search)
        document = {
            "@search.action": "upload",
            "id": chunk.get("id"),
            "content": chunk.get("content"),
            "contentVector": vector,
            "filepathField": chunk.get("filepathField"),
            "pagina": chunk.get("pagina"),
        }
        template["value"].append(document)
        batch_count += 1
        time.sleep(0.3)

        # Envio em Lotes
        if batch_count >= 10:
            uploadToIndex(index_full_name, template)
            batch_count = 0
            template = {"value": []}

    # Envia o lote final
    if template["value"]:
        uploadToIndex(index_full_name, template)

    return {"status": "Upload e vetorização concluídos."}

 
def extract_and_split_docs_v3(pdf_files: List[str], index_name: str) -> List[Dict[str, Any]]:

    """

    Processa PDFs, injeta metadados de contexto no conteúdo e divide em chunks 

    otimizados para tabelas, associando-os ao index_name fornecido.

    """

    # Splitter ajustado para tabelas: chunk maior e respeita quebra de linha

    text_splitter = RecursiveCharacterTextSplitter(

        chunk_size=2000,

        chunk_overlap=300,

        separators=["\n\n", "\n", " "], # Tenta manter as linhas da tabela juntas

        length_function=len,

    )

    chunks_to_upload = []
 
    for file_path in pdf_files:

        filename = os.path.basename(file_path)

        try:

            print(f"📄 Processando para o índice [{index_name}]: {filename}")

            # Importações locais conforme sua estrutura

            from documents_utils import extract_pages_from_pdf_simple_and_format_tables, clean_key

            pages_data = extract_pages_from_pdf_simple_and_format_tables(file_path, filename)

            for page_entry in pages_data:

                page_text = page_entry["text"]

                metadata = page_entry["metadata"]

                page_num = metadata.get("page_number", "N/A")

                # Divisão do texto da página em blocos menores

                texts = text_splitter.split_text(page_text)

                for i, chunk in enumerate(texts):

                    # Limpeza do nome do arquivo para compor o ID único

                    doc_id_base = clean_key(filename.replace(".pdf", "")).lower()

                    unique_id = f"{doc_id_base}-{page_num}-{i}"

                    # INJEÇÃO DE CONTEXTO: Fundamental para tabelas densas

                    # Isso garante que o modelo saiba a origem do dado mesmo em buscas granulares

                    content_enriquecido = f"FONTE: {filename} | PAGINA: {page_num} | INDICE: {index_name}\n---\n{chunk}"

                    chunks_to_upload.append({

                        "id": unique_id,

                        "content": content_enriquecido,

                        "filepathField": file_path,

                        "pagina": f"Página {page_num}",

                        "index_target": index_name # Metadado extra opcional para controle

                    })

        except Exception as e:

            print(f"❌ Erro crítico ao processar {filename}: {e}")

    return chunks_to_upload
 


if __name__ == "__main__":
    pdf_directory = "assets/documentos_pdf"

    index_name_target = "agente-assistente-virtual-sadt-3"

    pdf_files = glob(os.path.join(pdf_directory, "*.pdf"))

    if not pdf_files:
        print(f"❌ Erro: Nenhuma arquivo PDF encontrado no diretório: {pdf_directory}")
        # Você pode adicionar um 'exit()' aqui se não quiser prosseguir sem arquivos.
    else:
        print(f"✅ Encontrados {len(pdf_files)} arquivos PDF para processar.")
        # print("Lista de arquivos:", pdf_files) # Linha opcional para depuração

        # 3. Extrai e divide os documentos (agora usando a lista de caminhos)
        print(
            f"\nIniciando a extração e o splitting para o índice: {index_name_target}"
        )

        chunks = extract_and_split_docs_v3(
            pdf_files=pdf_files,  # Passamos a lista gerada dinamicamente
            index_name=index_name_target,
        )

        # 4. Vetoriza e faz o upload dos chunks
        print("\nIniciando vetorização e upload para o Azure AI Search...")
        vectorize_and_upload_v2(chunks, index_name_target)

        print("\nProcessamento finalizado.")
