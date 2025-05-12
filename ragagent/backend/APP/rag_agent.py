import os
import uuid
import tempfile
import logging
import json
import uuid
from uuid import uuid4
from datetime import datetime
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
7
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

            # Detect file type
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

            # ✅ Store in AstraDB
            if self.astra_db:
                astra_formatted_docs = []
                for i, chunk in enumerate(small_chunks):
                    metadata = chunk.metadata.copy() if chunk.metadata else {}
                    metadata.update({
                        "uploaded_by": user_context.get("name", "unknown"),
                        "source_filename": filename,
                        "domain": domain,
                        "chunk_index": i,
                        "filetype": "csv" if suffix == ".csv" else "text"
                    })

                    astra_formatted_docs.append(
                        Document(
                            page_content=chunk.page_content,
                            metadata=metadata
                        )
                    )

                await self.astra_db.aadd_documents(astra_formatted_docs)
                logger.info(f"✅ Stored {len(astra_formatted_docs)} chunks from file: {filename}")

            # ✅ Also store document summary in Cassandra for dashboards
            joined_content = "\n".join([doc.page_content for doc in small_chunks[:5]])[:3000]

            self.db.session.execute(
                """
                INSERT INTO documents (doc_id, filename, uploaded_by, domain, content, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    uuid4(),
                    filename,
                    user_context.get("name", "unknown"),
                    domain,
                    joined_content,
                    datetime.utcnow()
                )
            )
            logger.info(f"📦 Document metadata inserted into Cassandra for {filename}")

            return True

        except Exception as e:
            logger.error(f"❌ Document processing failed for {filename}: {str(e)}")
            return False

        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)



    async def generate_response(self, query: str, user_context: Optional[Dict] = None) -> Dict[str, any]:
        context_docs = []
        try:
            logger.debug("🔍 Starting response generation")
            logger.debug(f"Query received: {query}")

            # 1. Semantic Cache First (Double safety)
            cached = await self.cache.search(query)
            if cached:
                logger.info("✅ Cache HIT inside RAGAgent (unexpected fallback)")
                return {
                    "answer": cached,
                    "sources": [],
                    "confidence": 1.0,
                    "source_type": "cache"
                }

            # 2. Vector search setup
            if not self.astra_db:
                raise ValueError("❌ No vector store available")

            name = user_context.get("name") if user_context else None
            source_filename = user_context.get("source_filename") if user_context else None

            # STEP 1: Try file-specific search
            if source_filename:
                logger.info(f"🎯 Searching in file: {source_filename}")
                retriever = self.astra_db.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 5, "filter": {"source_filename": source_filename}}
                )
                context_docs = await retriever.ainvoke(query)

            # STEP 2: If no results, try global user files
            if not context_docs and name:
                logger.warning("⚠️ No results in file — searching across all user uploads.")
                retriever = self.astra_db.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 5, "filter": {"uploaded_by": name}}
                )
                context_docs = await retriever.ainvoke(query)

            # STEP 3: If still no results, fallback to LLM-only
            if not context_docs:
                logger.warning("🚨 No relevant content found. Falling back to LLM only.")
                prompt = f"You are a helpful assistant. Answer the following question:\n\n{query}"
                raw_answer_obj = await self.llm.generate(prompt)
                answer = self._extract_text_from_llm_output(raw_answer_obj)
                await self.cache.store(query, answer)

                return {
                    "answer": answer + "\n\n⚠️ *No relevant document found. This answer is based on general knowledge.*",
                    "sources": [],
                    "confidence": 0.6,
                    "source_type": "llm_no_context"
                }
            # Detect CSV file or question that needs table output
            is_csv = any(doc.metadata.get("filetype") == "csv" for doc in context_docs)
            user_query = query.lower()
            wants_table = any(keyword in user_query for keyword in ["compare", "comparison", "versus", "vs", "difference", "table"])

            # STEP 4: Build enhanced prompt with context
            prompt = self._build_enhanced_prompt(query, context_docs, user_context or {}, format_as_table=(is_csv or wants_table))
            logger.debug(f"Generated prompt preview:\n{prompt[:500]}...")

            raw_answer_obj = await self.llm.generate(prompt)
            answer = self._extract_text_from_llm_output(raw_answer_obj)
            await self.cache.store(query, answer)

            # STEP 5: Save history if session exists
            if user_context and "name" in user_context:
                try:
                    session_id = user_context.get("session_id")
                    self.db.save_chat_history(
                        username=user_context["name"],
                        question=query,
                        answer=answer,
                        session_id=session_id
                    )
                    logger.info(f"✅ Chat saved for {user_context['name']} in session {session_id}")
                except Exception as e:
                    logger.error(f"⚠️ Failed to save chat history: {e}")

            # STEP 6: Attach source info
            # Deduplicate by (filename, domain)
            seen_sources = set()
            source_snippets = []

            for doc in context_docs:
                meta = doc.metadata
                name = meta.get("source_filename", "unknown")
                domain = meta.get("domain", "unknown")
                key = (name, domain)
                if key not in seen_sources:
                    seen_sources.add(key)
                    source_snippets.append(f"📄 *{name}*  |  🧠 *{domain}*")

            source_footer = "\n\n---\n**Retrieved from:**\n" + "\n".join(source_snippets) if source_snippets else ""

            return {
                "answer": answer + source_footer,
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

    def _build_enhanced_prompt(self, query: str, context: List, user_context: Dict, format_as_table: bool = False) -> str:
        context_str = "\n\n".join(
            f"{doc.page_content}" for doc in context
        )

        file_info = ""
        if user_context and "source_filename" in user_context:
            file_info = f"from the file '{user_context['source_filename']}'"

        formatting_instructions = (
            "- If applicable, format your answer using a **Markdown table**.\n"
            "- Use rows and columns to clearly compare values or summarize information.\n"
            "- Use headings (#, ##) only when necessary.\n"
        ) if format_as_table else (
            "- Use headings (#, ##) where appropriate.\n"
            "- Use bullet points (-) for lists.\n"
            "- Keep paragraphs short and clear.\n"
        )

        return f"""
    You are a helpful AI Assistant.

    You have access to the following uploaded document content {file_info}:

    --- Document Content Start ---
    {context_str}
    --- Document Content End ---

    Rules:
    - ONLY use the provided document content to answer the question.
    - Quote text directly from the document when possible.
    - DO NOT guess or invent any new information not found in the document.
    - Write your answer in clean Markdown format:
    {formatting_instructions}
    - Do NOT use HTML or CSS formatting.

    User Question:
    {query}
    """

    def close(self):
        self.db.close()
        self.cache.clear_cache()
        if self.astra_db:
            self.astra_db.clear()
        logger.info("🔌 RAG agent shutdown complete")
