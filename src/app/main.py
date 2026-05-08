import os

import streamlit as st

st.set_page_config(
    page_title="Family Intelligence Vault",
    page_icon="🧠",
    layout="wide"
)

from core import RAGQueryService

DB_DIR = os.getenv("DB_DIR", "/app/chroma_db")
LLM_MODEL = "llama3.2"

st.title("🧠 Secure Knowledge Query")


@st.cache_resource
def get_service():
    return RAGQueryService(
        db_path=DB_DIR,
        llm_model=LLM_MODEL,
    )


service = get_service()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about your indexed documents..."):
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching vault..."):
            response = service.ask(prompt)

            answer = response["result"]
            sources = response.get("source_documents", [])

            st.markdown(answer)

            if sources:
                with st.expander("View Source Citations"):
                    for i, doc in enumerate(sources):
                        st.caption(
                            f"Source {i + 1}: {doc.metadata.get('source', 'Unknown')}"
                        )
                        st.info(doc.page_content)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
