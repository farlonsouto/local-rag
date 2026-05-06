from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings

from shared.config import Config


class RAGQueryService:
    """
    Business Logic for Retrieval Augmented Generation.
    Principle: Interface Segregation - provides a clean 'ask' method to the UI.
    """

    def __init__(self, db_path: str, llm_model: str):
        # Initialize the 'Librarian' (Embeddings)
        self.embeddings = OllamaEmbeddings(model=Config.EMBED_MODEL, base_url=Config.OLLAMA_BASE_URL)

        # Connect to the existing Vector Store (Read-Only)
        self.db = Chroma(persist_directory=db_path, embedding_function=self.embeddings)

        # Initialize the 'Thinker' (LLM)
        self.llm = ChatOllama(model=llm_model, base_url=Config.OLLAMA_BASE_URL, temperature=0)

    def _get_prompt_template(self) -> PromptTemplate:
        """Defines the 'System Instructions' for the LLM."""
        template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Keep the answer as concise as possible.

        {context}

        Question: {question}
        Helpful Answer:"""
        return PromptTemplate.from_template(template)

    def ask(self, query: str):
        """
        Executes the RAG cycle:
        1. Embeds query. 2. Retrieves context. 3. Augments prompt. 4. Generates answer.
        """
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.db.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": self._get_prompt_template()},
            return_source_documents=True
        )
        return qa_chain.invoke({"query": query})
