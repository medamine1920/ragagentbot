import os
import uuid
import tempfile
import logging
import json
import traceback
from typing import List, Dict, Optional
from pathlib import Path
from langchain_astradb import AstraDBVectorStore
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import UnstructuredFileLoader, PyPDFLoader

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
        self.hf_embedding = HuggingFaceEmbeddings(
            model_name="./models/e5-large-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.cache = SemanticCache(self.hf_embedding)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len
        )
        self.astra_db = None
        self.memory = ConversationBufferMemory(return_messages=True)

        self.setup_vector_stores()

    def setup_vector_stores(self):
        try:
            self.astra_db = AstraDBVectorStore(
                embedding=self.hf_embedding,
                collection_name="rag_collection_v2",
                api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
                token=os.getenv("ASTRA_DB_APPLICATION_TOKEN")
            )
            logger.info("✅ AstraDB vector store initialized successfully")
        except Exception as e:
            logger.warning(f"⚠️ AstraDB initialization failed: {e}")
            self.astra_db = None

    async def process_document(self, file_bytes: bytes, filename: str, domain: str, user_context: dict) -> bool:
        temp_path = None
        try:
            suffix = os.path.splitext(filename)[1].lower()
            logger.info(f"📄 Received file '{filename}' with suffix '{suffix}'")

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file_bytes)
                temp_path = tmp.name

            # Detect file type better
            if suffix == '.pdf':
                loader = PyPDFLoader(temp_path)
            elif suffix in ['.txt', '.md', '.csv']:
                loader = UnstructuredFileLoader(temp_path)
            else:
                logger.error(f"❌ Unsupported file type: {suffix}")
                return False

            # Load and split documents
            documents = loader.load()
            small_chunks = self.text_splitter.split_documents(documents)

            if not small_chunks:
                logger.warning(f"⚠️ No chunks created from {filename}. Skipping.")
                return False

            if self.astra_db:
                for chunk in small_chunks:
                    chunk.metadata["uploaded_by"] = user_context.get("email", "unknown")

                astra_formatted_docs = [
                    Document(page_content=chunk.page_content, metadata=chunk.metadata)
                    for chunk in small_chunks
                ]
                await self.astra_db.aadd_documents(astra_formatted_docs)

            logger.info(f"✅ Uploaded and processed {len(documents)} docs into {len(small_chunks)} chunks")
            return True

        except Exception as e:
            logger.error(f"❌ Document processing failed for {filename}: {str(e)}")
            return False

        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)


    async def generate_response(self, query: str, user_context: Optional[Dict] = None) -> Dict[str, any]:
        try:
            logger.debug("🔍 Starting response generation")
            logger.debug(f"Query received: {query}")

            # 1. Check Semantic Cache (already checked in /chat normally, but double-safety)
            cached = await self.cache.search(query)
            if cached:
                logger.info("✅ Cache HIT inside RAGAgent (unexpected fallback)")
                return {
                    "answer": cached,
                    "sources": [],
                    "confidence": 1.0,
                    "source_type": "cache"
                }

            # 2. Search in Astra Vector Store
            if not self.astra_db:
                raise ValueError("❌ No vector store available")
            
            retriever = self.astra_db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
            context_docs = await retriever.ainvoke(query)

            if not context_docs:
                logger.warning("⚠️ No documents found — falling back to pure Gemini LLM")
                prompt = f"You are a helpful assistant. Answer the following question:\n\n{query}"
                raw_answer_obj = await self.llm.generate(prompt)
                answer = self._extract_text_from_llm_output(raw_answer_obj)
                await self.cache.store(query, answer)

                return {
                    "answer": answer,
                    "sources": [],
                    "confidence": 0.7,
                    "source_type": "llm_no_context"
                }

            # 3. Build enhanced prompt with context
            prompt = self._build_enhanced_prompt(query, context_docs, user_context or {})
            logger.debug(f"Generated prompt (preview): {prompt[:500]}...")

            raw_answer_obj = await self.llm.generate(prompt)
            answer = self._extract_text_from_llm_output(raw_answer_obj)

            await self.cache.store(query, answer)

            if user_context and "name" in user_context:
                try:
                    session_id = user_context.get("session_id")  # 💡 get session_id from user_context
                    self.db.save_chat_history(
                        username=user_context["name"],
                        question=query,
                        answer=answer,
                        session_id=session_id
                    )
                    logger.info(f"✅ Chat saved for {user_context['name']} in session {session_id}")
                except Exception as e:
                    logger.error(f"⚠️ Failed to save chat history: {e}")


            return {
                "answer": answer,
                "sources": [doc.metadata for doc in context_docs],
                "confidence": 1.0,
                "source_type": "llm_with_context"
            }

        except Exception as e:
            logger.error(f"❌ Response generation failed: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "answer": f"<div class='error'>Internal Error: {str(e)}</div>",
                "sources": [],
                "confidence": 0.0,
                "source_type": "error"
            }

    def _extract_text_from_llm_output(self, raw_obj):
        text_attr = getattr(raw_obj, "text", None)
        if callable(text_attr):
            return text_attr().strip()
        elif isinstance(text_attr, str):
            return text_attr.strip()
        else:
            return str(raw_obj).strip()

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
                Respond naturally in clean Markdown format, with bullet points, headers if needed, and short paragraphs.
                Avoid using HTML or CSS.
.

{user_info}

CONTEXT:
{context_str}

QUESTION: {query}

Respond in clean Markdown format:
- Use headings (#, ##) where appropriate
- Use bullet points (-) if listing items
- Write short paragraphs
- Do not use raw HTML or inline CSS"""

    def close(self):
        self.db.close()
        self.cache.clear_cache()
        if self.astra_db:
            self.astra_db.clear()
        logger.info("🔌 RAG agent shutdown complete")
