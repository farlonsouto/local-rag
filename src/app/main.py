import os

import streamlit as st

from core import RAGQueryService

# Environment Config
DB_DIR = os.getenv("DB_DIR", "/app/chroma_db")
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
LLM_MODEL = "llama3.2"

st.set_page_config(page_title="Family Intelligence Vault", icon="🧠")
st.title("🧠 Secure Knowledge Query")


# Initialize Service (Cached to prevent reloading models on every click)
@st.cache_resource
def get_service():
    return RAGQueryService(
        db_path=DB_DIR,
        llm_model=LLM_MODEL,
        embed_model="nomic-embed-text",
        ollama_url=OLLAMA_URL
    )


service = get_service()

# Chat UI
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask about your documents (e.g. ID numbers, dates)..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching vault..."):
            response = service.ask(prompt)
            answer = response["result"]
            sources = response["source_documents"]

            st.markdown(answer)

            # Show Citations for auditability
            if sources:
                with st.expander("View Source Citations"):
                    for i, doc in enumerate(sources):
                        st.caption(f"Source {i + 1}: {doc.metadata.get('source', 'Unknown')}")
                        st.info(doc.page_content)

    st.session_state.messages.append({"role": "assistant", "content": answer})
