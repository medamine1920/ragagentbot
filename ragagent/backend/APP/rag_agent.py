import os
import uuid
import tempfile
import logging
import json
from langchain_astradb import AstraDBVectorStore

from typing import List, Dict, Optional
from pathlib import Path
from collections import OrderedDict
import time
import google.generativeai as genai

from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredFileLoader
from langchain.retrievers import (
    ContextualCompressionRetriever,
    MultiQueryRetriever,
    ParentDocumentRetriever,
    EnsembleRetriever
)
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import AstraDB
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from services.cassandra_connector import CassandraConnector
from services.llm_service import GeminiService
from services.semantic_cache import SemanticCache
from config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGAgent:
    def __init__(self):
        self.db = CassandraConnector()
        self.llm = GeminiService()
        self.cache = SemanticCache()

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len
        )

        self.hf_embedding = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        self.astra_db = None
        self.parent_store = InMemoryStore()
        self.memory = ConversationBufferMemory(return_messages=True)

        self.setup_vector_stores()

    def setup_vector_stores(self):
        try:
            self.astra_db = AstraDBVectorStore(
                embedding=self.hf_embedding,
                collection_name="rag_collection",
                api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
                token=os.getenv("ASTRA_DB_APPLICATION_TOKEN")
            )
            logger.info("✅ AstraDB vector store initialized successfully")
        except Exception as e:
            logger.warning(f"⚠️ AstraDB initialization failed: {e}")
            self.astra_db = None

    async def setup_ensemble_retrievers(self):
        if not self.astra_db:
            raise ValueError("❌ No vector store available")

        parent_retriever = self.configure_parent_child_splitters()
        retrieval = self.astra_db.as_retriever(
            search_type="mmr",
            search_kwargs={'k': 5, 'fetch_k': 50}
        )

        ensemble_retriever = EnsembleRetriever(
            retrievers=[parent_retriever, retrieval],
            weights=[0.4, 0.6]
        )

        multi_retriever = MultiQueryRetriever.from_llm(
            retriever=ensemble_retriever,
            llm=self.llm
        )

        return ContextualCompressionRetriever(
            base_compressor=None,
            base_retriever=multi_retriever
        )

    def configure_parent_child_splitters(self):
        parent_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=0)
        child_splitter = TokenTextSplitter(chunk_size=128, chunk_overlap=0)

        return ParentDocumentRetriever(
            vectorstore=self.astra_db,
            docstore=self.parent_store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
        )

    async def process_document(self, file_bytes: bytes, filename: str) -> bool:
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp:
                tmp.write(file_bytes)
                temp_path = tmp.name

            loader = PyPDFLoader(temp_path) if filename.lower().endswith('.pdf') else UnstructuredFileLoader(temp_path)
            documents = loader.load()

            small_chunks = self.text_splitter.split_documents(documents)
            if self.astra_db:
                print("📦 Chunk count:", len(small_chunks))
                print("🔍 Example chunk:\n", small_chunks[0].page_content[:500])
                await self.astra_db.aadd_documents(small_chunks)
                print("✅ Uploaded chunks to AstraDB")

            logger.info(f"📄 Processed {len(documents)} pages into {len(small_chunks)} chunks")
            return True
        except Exception as e:
            logger.error(f"❌ Document processing failed: {str(e)}")
            return False
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)

    async def generate_response(self, query: str, user_context: Optional[Dict] = None) -> Dict[str, any]:
        try:
            if not hasattr(self, 'compression_retriever'):
                self.compression_retriever = await self.setup_ensemble_retrievers()

            cached = await self.cache.search(query)
            if cached:
                return cached

            context = await self.compression_retriever.ainvoke(query)
            prompt = self._build_enhanced_prompt(query, context, user_context or {})
            raw_answer = await self.llm.generate(prompt)
            final_html = raw_answer

            await self.cache.store(query, final_html)
            return {
                "answer": final_html,
                "sources": [doc.metadata for doc in context],
                "confidence": "High"
            }
        except Exception as e:
            logger.error(f"❌ Response generation failed: {str(e)}")
            return {
                "answer": "<div class='error'>Sorry, something went wrong.</div>",
                "sources": [],
                "confidence": "Low"
            }

    def _build_enhanced_prompt(self, query: str, context: List, user_context: Dict) -> str:
        context_str = "\n\n".join(
            f"SOURCE: {doc.metadata.get('source', 'unknown')}\nCONTENT:\n{doc.page_content}\n"
            for doc in context
        )

        user_info = ""
        if user_context:
            user_info = f"\nUser Profile:\n- Name: {user_context.get('name', 'Unknown')}" \
                        f"\n- Role: {user_context.get('role', 'Unknown')}" \
                        f"\n- Preferences: {user_context.get('preferences', 'None')}"

        return f"""You are an expert assistant. Answer the question using the context below.
                Format your response in HTML with inline CSS for styling.

{user_info}

CONTEXT:
{context_str}

QUESTION: {query}

Respond with:
- Well-structured HTML (divs, sections, etc.)
- Clean, responsive styling
- Semantic markup where appropriate
- No <html> or <head> tags, just <body> content"""

    def close(self):
        self.db.close()
        self.cache.clear_cache()
        if self.astra_db:
            self.astra_db.clear()
        logger.info("🔌 RAG agent shutdown complete")
