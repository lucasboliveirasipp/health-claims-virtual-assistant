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

import os
from typing import Dict, Any, List
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
    SystemMessage,
    RemoveMessage,
)
from langchain_core.prompts import PromptTemplate
from langchain_openai import (
    AzureOpenAIEmbeddings,
)  # Mantenha esta importação (se usar o LangChain)
from langchain_openai import AzureChatOpenAI
from .agent_state import (
    AgentState,
    GradeDocuments,
)  # Assumindo AgentState e GradeDocuments estão aqui
from .tools import SADT_retriever_tool  # Assumindo a ferramenta de busca está aqui
from rank_bm25 import BM25Okapi
import numpy as np
from .prompts import agent_prompt, grader_prompt, prompt_SADT

llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
    openai_api_version="2024-08-01-preview",
    deployment_name="gpt-4o-mini",
    openai_api_type="azure",
    temperature=0,
)


# -------------------
# CustomToolNode (Com extração de query robusta)
# -------------------
class CustomToolNode:

    def __init__(self, tools: list):

        self.tools_by_name = {t.name: t for t in tools}
 
    def __call__(self, inputs: dict):

        messages = inputs.get("messages", [])

        last_message = messages[-1]

        output_messages = []

        cumulative_docs = []

        final_query = ""

        final_index = inputs.get("index", "default")
 
        # Processa TODAS as tool_calls da mensagem

        for tool_call in last_message.tool_calls:

            tool_name = tool_call["name"]

            tool_args = tool_call["args"]

            # 1. Executa a ferramenta

            result = self.tools_by_name[tool_name].invoke(tool_args)

            # 2. Coleta documentos e query

            cumulative_docs.extend(result.get("docs", []))

            final_index = result.get("index", final_index)

            final_query = tool_args.get("query", "Busca")
 
            # 3. Cria a ToolMessage para CADA ID solicitado

            output_messages.append(

                ToolMessage(

                    content=f"Resultado da busca para '{final_query}' concluído.",

                    tool_call_id=tool_call["id"]

                )

            )
 
        # Retorna todas as ToolMessages de uma vez

        return {

            "query": final_query,

            "docs": cumulative_docs,

            "index": final_index,

            "messages": output_messages, # Aqui vão todas as respostas

        }
 


# -------------------
# Busca híbrida BM25 + semântica (CORRIGIDA - Recebe o modelo como argumento)
# -------------------
def hybrid_retriever(
    query: str, docs: list, embedding_model, alpha: float = 0.8, top_k: int = 10
):
    """
    Implementa re-ranqueamento BM25 + Semântico.
    O 'embedding_model' deve ser injetado para vetorizar a query.
    """
    if not docs:
        return []
    tokenized_corpus = [d.page_content.split() for d in docs]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(query.split())
    # Obtenção do embedding real da query
    if embedding_model:
        try:
            query_emb = np.array(embedding_model.embed_query(query))
        except Exception as e:
            print(f"Erro ao vetorizar query ({e}), usando array zero como fallback.")
            query_emb = np.zeros(1536)
    else:
        query_emb = np.zeros(1536)  # Fallback
    # Tenta obter embeddings dos documentos ou usa zeros
    doc_embs = []
    for d in docs:
        emb = getattr(d, "embedding", np.zeros(query_emb.shape[0]))
        if isinstance(emb, list):
            emb = np.array(emb)
        doc_embs.append(emb)

    doc_embs_array = np.array(doc_embs)
    # Normalização dos scores BM25
    bm25_max = np.max(bm25_scores)
    bm25_scores = bm25_scores / bm25_max if bm25_max > 0 else bm25_scores
    # Normalização dos scores Semânticos
    if (
        doc_embs_array.ndim == 2
        and doc_embs_array.shape[1] == query_emb.shape[0]
        and not np.all(doc_embs_array == 0)
    ):
        # Calcula a similaridade do cosseno
        semantic_scores = doc_embs_array @ query_emb
        semantic_scores_min = np.min(semantic_scores)
        semantic_scores_max = np.max(semantic_scores)

        if semantic_scores_max > semantic_scores_min:
            semantic_scores = (semantic_scores - semantic_scores_min) / (
                semantic_scores_max - semantic_scores_min
            )
        else:
            semantic_scores = np.zeros(len(docs))
    else:
        semantic_scores = np.zeros(len(docs))
    # Combinação dos scores
    combined_scores = alpha * bm25_scores + (1 - alpha) * semantic_scores
    top_indices = np.argsort(combined_scores)[::-1][:top_k]
    return [docs[i] for i in top_indices]


# -------------------
# Templates e Chains
# -------------------
grader_template = PromptTemplate(
    input_variables=["question", "messages", "document"], template=grader_prompt
)
llm_grader = llm.with_structured_output(GradeDocuments)
retrieval_grader = grader_template | llm_grader
agent_msg = PromptTemplate(input_variables=["index", "messages"], template=agent_prompt)
tools = [SADT_retriever_tool]
llm_with_tools = llm.bind_tools(tools)
tool_node = CustomToolNode(tools)


# -------------------
# Nodes (Funções do Grafo)
# -------------------
def should_summarize(state: AgentState):
    """Decide se deve resumir a conversa."""
    return "summarize_conversation" if len(state["messages"]) > 5 else "agent"


def summarize_conversation(state: AgentState):
    """Gera ou atualiza o resumo da conversa e limpa o histórico."""
    summary = state.get("summary", "")
    summary_prompt = (
        f"Resumo atual da conversa: {summary}\n\n"
        "Atualize o resumo considerando as novas mensagens, mantendo informações importantes e códigos citados."
        if summary
        else "Crie um resumo da conversa acima."
    )
    # Prepara o input para o llm de resumo
    messages_for_summary_llm = state["messages"] + [
        HumanMessage(content=summary_prompt)
    ]
    response = llm.invoke(messages_for_summary_llm)

    # 1. Mensagens a serem removidas (tudo exceto as 5 últimas)
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-6]]
    # 2. Mensagens a serem mantidas (as 5 últimas mensagens, que são as mais recentes)
    messages_to_keep = state["messages"][-6:]
    # 3. Retorna a instrução de remoção + as mensagens a manter no estado
    # O LangGraph irá remover as antigas e manter/adicionar as mensagens_to_keep
    return {
        "summary": response.content,
        # A nova lista de 'messages' para o estado será a fusão do 'RemoveMessage' e 'messages_to_keep'
        "messages": delete_messages + messages_to_keep,
    }


# Nó 'crag'
def grade_documents(state: AgentState):

    query = state["query"]

    mensagens = state.get("messages", [])

    docs_recuperados = state.get("docs", [])

    current_index = state["index"]

    embedding_model = state.get("embedding_model")
 
    if not docs_recuperados:

        return {"docs": [], "query": query, "index": current_index}
 
    # --- MELHORIA: Limpeza da Query para o BM25 ---

    # Remove palavras comuns que "sujam" a busca por termos técnicos

    stop_words = ["para", "o", "existe", "algum", "procedimento", "ou", "pacote", "negociado", "do", "da", "de", "com", "em"]

    clean_query_words = [w for w in query.lower().split() if w not in stop_words]

    clean_query = " ".join(clean_query_words)

    # 1. Re-ranqueamento Híbrido com a query limpa e Alpha 0.8

    filtered_docs = hybrid_retriever(clean_query, docs_recuperados, embedding_model, alpha=0.8)
 
    # 2. CRAG (Filtro via LLM)

    final_docs = []

    for d in filtered_docs:

        # Usamos a query original aqui para o LLM entender a intenção completa

        score = retrieval_grader.invoke(

            {"question": query, "document": d.page_content, "messages": mensagens}

        )

        if score.binary_score.lower().strip() == "yes":

            final_docs.append(d)

    # DEBUG atualizado para ver a query limpa

    print(f"DEBUG: Query original: {query}")

    print(f"DEBUG: Query limpa usada no BM25: {clean_query}")

    print(f"DEBUG: Docs após CRAG: {len(final_docs)}")

    return {"docs": final_docs, "query": query, "index": current_index}
 


# nodes.py - NOVO CÓDIGO para agent
def agent(state: AgentState):
    """
    Nó principal de raciocínio. Decide se chama a ferramenta ou gera a resposta.
    """
    current_index = state.get("index", "SADT")
    summary_text = state.get("summary", "")
    # 1. Pega as mensagens do estado (já limpas se o summarize rodou)
    messages_for_llm = state["messages"]
    # 2. Injeção segura do Resumo como SystemMessage (APENAS para o llm)
    if summary_text:
        summary_message = SystemMessage(
            content=f"Resumo da Conversa (Contexto para Raciocínio): {summary_text}"
        )
        # Coloca a mensagem de sistema no início da lista de input do llm
        messages_for_llm = [summary_message] + messages_for_llm
    agent_chain = agent_msg | llm_with_tools
    # 3. Retorna APENAS a nova mensagem do agente.
    return {
        "messages": [
            agent_chain.invoke({"index": current_index, "messages": messages_for_llm})
        ]
    }


def should_continue(state: AgentState):
    """Decide se a execução continua chamando a ferramenta ou termina."""
    last_message = state["messages"][-1]
    return "continue" if last_message.tool_calls else "end"


def generate(state: AgentState):
    """Gera a resposta final usando os documentos filtrados."""
    docs = state.get("docs", [])
    mensagens = state.get("messages", [])
    index = state.get("index", None)
    query = state.get("query", "")

    # Converte documentos para uma string limpa para o prompt
    docs_string = "\n---\n".join([f"Documento: {d.page_content}" for d in docs])

    # Cláusula de segurança unificada (conforme otimização do prompt)
    no_context_msg = "Desculpe, não consegui confirmar esta informação nos documentos da Acme Saúde."
    # Lógica de verificação da última pergunta (mantida)
    last_question_synonyms = [
        "qual foi a minha última pergunta",
        "qual era a minha última pergunta",
        "o que eu perguntei antes",
        "minha pergunta anterior",
        "qual foi minha pergunta anterior",
    ]
    if any(syn in query.lower() for syn in last_question_synonyms):
        last_valid_question = None
        for msg in reversed(mensagens):
            if isinstance(msg, HumanMessage):
                if not any(s in msg.content.lower() for s in last_question_synonyms):
                    last_valid_question = msg.content
                    break
        reply = (
            f'Sua última pergunta válida foi: "{last_valid_question}"'
            if last_valid_question
            else no_context_msg
        )
        return {"messages": [AIMessage(content=reply)], "docs": docs}
    # Caso existam documentos relevantes

    valid_indices = [
        "agente-assistente-virtual-sadt-3-index",
        "agente-assistente-virtual-sadt-index",
    ]

    if docs_string:
        result = None

        index_list = index if isinstance(index, list) else [index]
        # Verifica se pelo menos um dos índices retornados pela ferramenta está na lista de índices válidos
        if any(
            i in valid_indices for i in index_list
        ):  # Verifique se esta é a chave correta
            refund_template = PromptTemplate(
                input_variables=["question", "context", "messages"],
                template=prompt_SADT,
            )
            chain_refund = refund_template | llm
            result = chain_refund.invoke(
                {"question": query, "context": docs_string, "messages": mensagens}
            )
        else:
            text = f"Encontrei documentos no índice '{index}', mas não há uma lógica de resposta (template de prompt) definida para ele."
            return {"messages": [AIMessage(content=text)], "docs": docs}
        if result:
            text = result.content if isinstance(result, AIMessage) else str(result)
            return {"messages": [AIMessage(content=text)], "docs": docs}
        else:
            # Se o llm não conseguir gerar uma resposta com o contexto, usa a cláusula de segurança
            return {"messages": [AIMessage(content=no_context_msg)], "docs": docs}
    # Pergunta fora de contexto (sem docs)
    else:
        return {"messages": [AIMessage(content=no_context_msg)], "docs": docs}
