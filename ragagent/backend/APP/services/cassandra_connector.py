from cassandra.cluster import Cluster
from cassandra.policies import DCAwareRoundRobinPolicy
from cassandra.auth import PlainTextAuthProvider
from cassandra.query import dict_factory
import time
import logging

class CassandraConnector:
    def __init__(self, hosts=['cassandra'], port=9042, max_retries=5, retry_delay=5):
        self.cluster = None
        self.session = None
        self.connected = False
        
        # Configure load balancing policy
        lb_policy = DCAwareRoundRobinPolicy(local_dc='datacenter1')
        
        # Configure auth provider (even if not using auth)
        auth_provider = PlainTextAuthProvider(username='cassandra', password='cassandra')
        
        for attempt in range(max_retries):
            try:
                self.cluster = Cluster(
                    hosts,
                    port=port,
                    load_balancing_policy=lb_policy,
                    auth_provider=auth_provider,
                    protocol_version=4
                )
                self.session = self.cluster.connect()
                self.session.row_factory = dict_factory
                self._initialize_schema()
                self.connected = True
                logging.info("Successfully connected to Cassandra")
                break
            except Exception as e:
                logging.warning(f"Connection attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    raise

    def _initialize_schema(self):
        # Add retries for schema initialization
        self.session.execute("""
            CREATE KEYSPACE IF NOT EXISTS rag_demo 
            WITH replication = {
                'class': 'SimpleStrategy',
                'replication_factor': 1
            }
        """, timeout=60)  # Increased timeout
        
        self.session.set_keyspace('rag_demo')
        
        self.session.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id uuid PRIMARY KEY,
                content text,
                embedding list<float>,
                metadata map<text, text>
            )
        """, timeout=60)