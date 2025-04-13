import faiss
import numpy as np
from services.cassandra_connector import CassandraConnector
from services.llm_service import GeminiService
from datetime import datetime
import uuid
import logging
from typing import Optional  # ✅ Correct

from config import settings

logger = logging.getLogger(__name__)



class SemanticCache:
    def __init__(self):
        self.llm = GeminiService()
        self.db = CassandraConnector()
        self.index = faiss.IndexFlatL2(768)  # Google's embedding dimension
        self.id_map = {}
        self._load_cache()

    def _load_cache(self):
        """Load existing cache entries"""
        try:
            rows = self.db.session.execute(
                "SELECT doc_id, embedding FROM documents WHERE metadata['type'] = 'cache'"
            )
            embeddings = []
            for idx, row in enumerate(rows):
                embedding = np.array(row.embedding, dtype='float32')
                embeddings.append(embedding)
                self.id_map[idx] = str(row.doc_id)
            
            if embeddings:
                self.index.add(np.vstack(embeddings))
            logger.info(f"Loaded {len(embeddings)} cache entries")
        except Exception as e:
            logger.error(f"Cache loading failed: {str(e)}")

    async def search(self, query: str) -> Optional[str]:
        """Search for similar cached responses"""
        try:
            embedding = await self.llm.embed_text(query)
            embedding = np.array(embedding, dtype='float32').reshape(1, -1)
            
            distances, indices = self.index.search(embedding, 1)
            if indices[0][0] >= 0 and distances[0][0] <= settings.CACHE_THRESHOLD:
                doc_id = self.id_map[indices[0][0]]
                row = self.db.session.execute(
                    "SELECT content FROM documents WHERE doc_id = %s",
                    (uuid.UUID(doc_id),)
                ).one()
                return row.content if row else None
            return None
        except Exception as e:
            logger.error(f"Cache search failed: {str(e)}")
            return None

    async def store(self, content: str):
        """Store new response in cache"""
        try:
            embedding = await self.llm.embed_text(content)
            doc_id = uuid.uuid4()
            
            self.db.session.execute(
                """
                INSERT INTO documents 
                (doc_id, content, embedding, metadata)
                VALUES (%s, %s, %s, %s)
                """,
                (doc_id, content, embedding, 
                 {"type": "cache", "timestamp": str(datetime.now())})
            )
            
            self.index.add(np.array([embedding], dtype='float32'))
            self.id_map[self.index.ntotal - 1] = str(doc_id)
            logger.debug(f"Cached new response: {content[:50]}...")
        except Exception as e:
            logger.error(f"Cache store failed: {str(e)}")