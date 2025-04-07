import ollama
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredFileLoader
from services.cassandra_connector import CassandraConnector
from cassandra.cluster import Cluster
from cassandra.query import dict_factory
import numpy as np
import tempfile
import os
from typing import List, Dict, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGAgent:
    def __init__(self):
        self.db = CassandraConnector()
        # Model configurations (hardcoded)
        self.llm_model = "llama3"  # Options: llama3, mistral, phi3, etc.
        self.embedding_model = "all-MiniLM-L6-v2"  # Local embedding model
        
        # Initialize components
        self.embedder = SentenceTransformer(self.embedding_model)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        # Initialize Cassandra connection
        self.cluster = Cluster(['cassandra'])
        self.session = self.cluster.connect()
        self.session.set_keyspace('rag_demo')
        self.session.row_factory = dict_factory
        
        # Verify Ollama model is available
        self._verify_ollama_model()

    def _verify_ollama_model(self):
        """Ensure the specified Ollama model is available"""
        try:
            ollama.show(self.llm_model)
            logger.info(f"Ollama model '{self.llm_model}' is ready")
        except Exception as e:
            logger.warning(f"Model {self.llm_model} not found. Pulling...")
            ollama.pull(self.llm_model)

    async def process_document(self, file_bytes: bytes, filename: str) -> bool:
        """Process and store uploaded documents"""
        temp_path = None
        try:
            # Save to temp file for processing
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp:
                tmp.write(file_bytes)
                temp_path = tmp.name

            # Load document based on file type
            if filename.lower().endswith('.pdf'):
                loader = PyPDFLoader(temp_path)
            else:
                loader = UnstructuredFileLoader(temp_path)
            
            documents = loader.load()
            chunks = self.text_splitter.split_documents(documents)
            
            # Store each chunk in Cassandra
            for chunk in chunks:
                embedding = self.embedder.encode(chunk.page_content)
                
                self.session.execute(
                    """
                    INSERT INTO documents 
                    (doc_id, content, embedding, metadata)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (str(uuid.uuid4()), 
                    chunk.page_content,
                    embedding.tolist(),
                    {"source": filename, "type": "text_chunk"}
                ))
            
            logger.info(f"Processed {len(chunks)} chunks from {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to process {filename}: {str(e)}")
            return False
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)

    async def retrieve_context(self, query: str, k: int = 3) -> List[Dict]:
        """Retrieve relevant context chunks using vector similarity"""
        try:
            query_embedding = self.embedder.encode(query).tolist()
            
            # ANN search in Cassandra
            rows = self.session.execute(
                """
                SELECT content, metadata, similarity_cosine(embedding, %s) AS score
                FROM documents
                ORDER BY embedding ANN OF %s
                LIMIT %s
                """,
                (query_embedding, query_embedding, k))
            
            return [dict(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Retrieval failed: {str(e)}")
            return []

    async def generate_response(self, query: str) -> Dict[str, any]:
        """Generate LLM response with RAG"""
        try:
            # Retrieve relevant context
            context_chunks = await self.retrieve_context(query)
            context = "\n\n".join([chunk['content'] for chunk in context_chunks])
            
            # Construct prompt
            prompt = f"""Answer the question based on the following context:
            
            Context:
            {context}
            
            Question: {query}
            
            Answer:"""
            
            # Get LLM response
            response = ollama.generate(
                model=self.llm_model,
                prompt=prompt,
                stream=False,
                options={
                    "temperature": 0.7,
                    "num_ctx": 4096
                }
            )
            
            return {
                "answer": response['response'],
                "sources": [chunk['metadata'] for chunk in context_chunks],
                "from_cache": False
            }
            
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            return {
                "answer": "Sorry, I encountered an error processing your request.",
                "sources": [],
                "from_cache": False
            }

    def close(self):
        """Clean up resources"""
        self.session.shutdown()
        self.cluster.shutdown()
        logger.info("Closed Cassandra connection")