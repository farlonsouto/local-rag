import streamlit as st

from filewalker.core import DocumentIndexer

st.set_page_config(page_title="Filewalker | Indexer", icon="📂")

st.title("📂 Filewalker: Vector DB Builder")
st.markdown("""
This module performs **Semantic Indexing**. It reads your local PDFs, converts them 
into mathematical vectors, and stores them in the persistent storage.
""")

if st.button("🔄 Start Synchronization", use_container_width=True):
    bar = st.progress(0)
    label = st.empty()


    def update_ui(progress: float, text: str):
        bar.progress(progress)
        label.info(text)


    indexer = DocumentIndexer()

    try:
        indexer.run_sync(progress_fn=update_ui)
        st.success("The Vector Database is now up-to-date and ready for queries.")
        st.balloons()
    except Exception as e:
        st.error(f"Sync failed: {str(e)}")
        st.warning("Ensure the 'ollama' container is healthy and the GPU is available.")
