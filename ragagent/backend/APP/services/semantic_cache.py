import faiss
import numpy as np
from services.cassandra_connector import CassandraConnector
from sentence_transformers import SentenceTransformer
from config import Config
from datetime import datetime
import ollama
import uuid

class SemanticCache:
    def __init__(self):
        self.cassandra = CassandraConnector()
        self.embedder = SentenceTransformer(Config.EMBEDDING_MODEL)  
        self.index = faiss.IndexFlatL2(768)
        self._load_existing_cache()

    def _load_existing_cache(self):
        rows = self.cassandra.session.execute("SELECT doc_id, embedding FROM documents")
        embeddings = []
        self.id_map = {}
        
        for idx, row in enumerate(rows):
            embedding = np.array(row.embedding, dtype='float32')
            embeddings.append(embedding)
            self.id_map[idx] = row.doc_id
        
        if embeddings:
            self.index.add(np.vstack(embeddings))

    async def search(self, query):
        query_embedding = await self.embeddings.embed_query(query)
        query_embedding = np.array(query_embedding, dtype='float32')
        
        D, I = self.index.search(np.expand_dims(query_embedding, 0), 1)
        
        if I[0][0] >= 0 and D[0][0] <= Config.CACHE_THRESHOLD:
            doc_id = self.id_map[I[0][0]]
            row = self.cassandra.session.execute(
                "SELECT content FROM documents WHERE doc_id = %s",
                (doc_id,)
            ).one()
            return row.content if row else None
        return None

    async def store(self, content):
        embedding = await self.embeddings.embed_query(content)
        doc_id = self.cassandra.store_document(
            content=content,
            embedding=list(embedding),
            metadata={"type": "cache", "timestamp": str(datetime.now())}
        )
        
        # Add to FAISS index
        self.index.add(np.array([embedding], dtype='float32'))
        self.id_map[self.index.ntotal - 1] = doc_id