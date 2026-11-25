from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from uuid import uuid1
import streamlit as st
import time
from .graph import graph
from datetime import datetime
# Mantendo suas importações de banco
from feedback_local import MONGO
from feedback_local import get_local_datetime_for_mongo
from feedback_local import log_interaction
from uuid import uuid4
 
AGENT_NAME = "agente_assistente_virtual"
 
def initialize_chat_state():
    """Inicializa os estados necessários no 'st.session_state'"""
    if f"{AGENT_NAME}_chat_history" not in st.session_state:
        st.session_state[f"{AGENT_NAME}_chat_history"] = []
    if f"{AGENT_NAME}_thread_id" not in st.session_state:
        st.session_state[f"{AGENT_NAME}_thread_id"] = str(uuid4())
def reset_chat_state():
    """Reseta o estado do chat e gera um novo thread_id"""
    st.session_state[f"{AGENT_NAME}_chat_history"] = []
    st.session_state[f"{AGENT_NAME}_thread_id"] = str(uuid4())
def run(text, config, option):  
    return graph.invoke({"messages": text, "index": option}, config)  
 
def save_feedback_db(score: int, msg: dict):
    """Lógica de salvamento igual ao ouvidoria: via MongoDB direto"""
    thread_id = st.session_state[f"{AGENT_NAME}_thread_id"]
    try:
        collection = MONGO[AGENT_NAME]
        feedback_entry = {
            "timestamp": get_local_datetime_for_mongo(),
            "response_id": msg.get("response_id"),
            "chatbot_response": msg.get("content"),
            "user_question": msg.get("user_query"),
            "feedback": score,
        }
        collection.update_one(
            {"_id": thread_id},
            {
                "$set": {
                    "agent_name": AGENT_NAME,
                    "datetime": get_local_datetime_for_mongo(),
                },
                "$push": {"interactions": feedback_entry},
            },
            upsert=True,
        )
        st.toast("✅ Feedback salvo!")
    except Exception as e:
        st.error(f"Erro ao salvar: {e}")
 
def build_page(is_authenticated: bool):
    initialize_chat_state()
    if is_authenticated:  
        st.title("Assistente virtual sinistro de rede")
        option = st.selectbox("Escolha a area que deseja realizar a consulta:", options=["SADT"])
        if f"{AGENT_NAME}_chat_history" not in st.session_state:
            st.session_state[f"{AGENT_NAME}_chat_history"] = []
 
        # Mantendo sua estrutura de layout exatamente igual
        conversational = st.columns((1, 14))
        output = st.empty()
        message_container = st.container()  
        input_container = st.container()
        with input_container:
            conversational = st.columns((1, 14))
            if conversational[0].button(label="🗑️"):
                reset_chat_state()
                st.rerun()
 
            prompt = conversational[1].chat_input("Digite sua consulta:")
 
        if prompt:
            st.session_state[f"{AGENT_NAME}_chat_history"].append({"role": "user", "content": prompt})
 
            config = {"configurable": {"thread_id": st.session_state[f"{AGENT_NAME}_thread_id"]}}
 
            try:
                response = run(prompt, config, option)
                res_content = response["messages"][-1].content
                res_id = str(uuid4())
 
                # Salvamos a resposta com o ID e a pergunta para o feedback futuro
                st.session_state[f"{AGENT_NAME}_chat_history"].append({
                    "role": "assistant", 
                    "content": res_content, 
                    "docs": response.get("docs", None),
                    "response_id": res_id,
                    "user_query": prompt
                })
                log_interaction(
                    agent_name=AGENT_NAME,
                    thread_id=st.session_state[f"{AGENT_NAME}_thread_id"],
                    response_id=res_id,
                    user_question=prompt,
                    chatbot_response=res_content,
                    docs=response.get("docs", None)
                )                            
                st.rerun()
 
            except Exception as e:
                st.error(f"Erro durante a execução: {e}")
        with message_container:
            # Mantendo seu CSS original
            st.markdown("""
<style>
                    .stContainer {
                        max-height: 800px;
                        overflow-y: auto;
                        padding-right: 100px;
                    }
</style>
            """, unsafe_allow_html=True)
            for index, msg in enumerate(st.session_state[f"{AGENT_NAME}_chat_history"]):
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    # Feedback igual ao ouvidoria
                    if msg["role"] == "assistant" and "response_id" in msg:
                        res_id = msg["response_id"]
                        # Widget de thumbs
                        score = st.feedback("thumbs", key=f"fb_{res_id}")
                        if score is not None:
                            # Chave para evitar salvar múltiplas vezes o mesmo feedback
                            already_saved = f"saved_{res_id}"
                            if not st.session_state.get(already_saved, False):
                                save_feedback_db(score, msg)
                                st.session_state[already_saved] = True
 
    else:
        st.error("Você não está autenticado. Por favor, faça login para acessar esta página.")