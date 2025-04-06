from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
import uuid
from datetime import datetime
#from config import Config

class CassandraConnector:
    def __init__(self):
        # Docker-specific configuration
        self.cluster = Cluster(
            contact_points=['great_jennings'],  # Container name
            port=9042,  # Host-mapped port
            protocol_version=4,
            auth_provider=PlainTextAuthProvider(
                username='cassandra',
                password='cassandra'  # Default credentials
            )
        )
        self.session = self.cluster.connect()
        self._initialize_schema()

    def _initialize_schema(self):
        self.session.execute("""
            CREATE KEYSPACE IF NOT EXISTS rag_demo 
            WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1}
        """)
        self.session.set_keyspace('rag_demo')
        
        self.session.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id UUID PRIMARY KEY,
                content TEXT,
                embedding VECTOR<FLOAT, 768>,
                metadata MAP<TEXT, TEXT>
            )
        """)
        
        # Other table creations (users, chat_history, etc.)
        # ...

    def store_document(self, content, embedding, metadata):
        doc_id = uuid.uuid4()
        self.session.execute(
            """
            INSERT INTO documents (doc_id, content, embedding, metadata)
            VALUES (%s, %s, %s, %s)
            """,
            (doc_id, content, embedding, metadata)
        )
        return doc_id

    def close(self):
        self.cluster.shutdown()