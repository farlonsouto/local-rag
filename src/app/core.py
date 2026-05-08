from pathlib import Path

from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings

from shared.config import Config


class RAGQueryService:
    """
    Retrieval-Augmented Generation service layer.
    Keeps UI separated from retrieval and LLM orchestration.
    """

    def __init__(self, db_path: str, llm_model: str):
        Path(db_path).mkdir(parents=True, exist_ok=True)

        self.embeddings = OllamaEmbeddings(
            model=Config.EMBED_MODEL,
            base_url=Config.OLLAMA_BASE_URL,
        )

        self.db = Chroma(
            persist_directory=db_path,
            embedding_function=self.embeddings,
        )

        self.llm = ChatOllama(
            model=llm_model,
            base_url=Config.OLLAMA_BASE_URL,
            temperature=0,
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.db.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": self._get_prompt_template()},
            return_source_documents=True,
        )

    def _get_prompt_template(self) -> PromptTemplate:
        template = """
Use the following context to answer the question.

If the answer is unknown, say you don't know.
Do not invent facts.
Keep answers concise.

Context:
{context}

Question:
{question}

Answer:
"""
        return PromptTemplate.from_template(template)

    def ask(self, query: str):
        return self.qa_chain.invoke({"query": query})
