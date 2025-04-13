from collections import defaultdict
from cassandra.cluster import Cluster
import cassio
import uuid
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
import google.generativeai as genai
import logging

load_dotenv()

logger = logging.getLogger(__name__)

def categorize_timestamp(ts: Optional[datetime]) -> Optional[str]:
    """Categorize timestamp into relative time periods"""
    if ts is None:
        return None
        
    now = datetime.now()
    today = now.date()
    session_date = ts.date()

    if session_date == today:
        return "Today"
    elif session_date == today - timedelta(days=1):
        return "Yesterday"
    elif session_date >= today - timedelta(days=7):
        return "Last 7 Days"
    elif session_date >= today - timedelta(days=30):
        return "Last 30 Days"
    else:
        return session_date.strftime("%B %Y")

class CassandraConnector:
    """Enhanced Cassandra connector with session management, user operations,
    and chat history tracking, following the structure of CassandraManager.
    """
    
    def __init__(self):
        self.CASSANDRA_HOST = os.getenv("CASSANDRA_HOST", "cassandra")
        self.CASSANDRA_PORT = int(os.getenv("CASSANDRA_PORT", "9042"))
        self.KEYSPACE = os.getenv("CASSANDRA_KEYSPACE", "rag_keyspace")

        self.cluster = Cluster(contact_points=[self.CASSANDRA_HOST], port=self.CASSANDRA_PORT)
        self.session = self._initialize_session()
        self._initialize_schema()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources on context exit"""
        if exc_type:
            logger.error(f"Cassandra error: {exc_val}", exc_info=True)
        self.session.shutdown()

    def _initialize_session(self):
        """Initialize Cassandra session with connection pooling"""
        cluster = Cluster(
            [self.CASSANDRA_HOST],
            port=int(self.CASSANDRA_PORT),
            protocol_version=4,
            idle_heartbeat_interval=30
        )
        session = cluster.connect()
        session.set_keyspace(self.KEYSPACE)  # ✅ Add this line
        cassio.init(session=session, keyspace=self.KEYSPACE)
        return session

    def _initialize_schema(self):
        """Create all required tables if they don't exist"""
        self._execute_cql(
            f"""
            CREATE KEYSPACE IF NOT EXISTS {self.KEYSPACE}
            WITH replication = {{'class': 'SimpleStrategy', 'replication_factor': 1}}
            """
        )
        
        # Create tables with improved schema
        tables = {
            "vectores": """
                CREATE TABLE IF NOT EXISTS {keyspace}.vectores (
                    partition_id UUID PRIMARY KEY,
                    document_text TEXT,
                    document_content BLOB,
                    vector BLOB
                )
            """,
            "users_new": """
                CREATE TABLE IF NOT EXISTS {keyspace}.users_new (
                    user_id UUID PRIMARY KEY,
                    username TEXT,
                    email TEXT,
                    his_job TEXT,
                    password TEXT
                )
            """,
            "session_table": """
                CREATE TABLE IF NOT EXISTS {keyspace}.session_table (
                    session_id UUID,
                    title TEXT,
                    timestamp TIMESTAMP,
                    user_id UUID,
                    PRIMARY KEY (session_id)
                )
            """,
            "response_session": """
                CREATE TABLE IF NOT EXISTS {keyspace}.response_session (
                    partition_id UUID PRIMARY KEY,
                    session_id UUID,
                    table_response_id UUID
                )
            """,
            "response_table": """
                CREATE TABLE IF NOT EXISTS {keyspace}.response_table (
                    partition_id UUID PRIMARY KEY,
                    question TEXT,
                    answer TEXT,
                    timestamp TIMESTAMP,
                    evaluation BOOLEAN
                )
            """
        }

        for table, cql in tables.items():
            self._execute_cql(cql.format(keyspace=self.KEYSPACE))

    def _execute_cql(self, query: str, params: tuple = None) -> Any:
        """Execute CQL query with error handling"""
        try:
            if params:
                return self.session.execute(query, params)
            return self.session.execute(query)
        except Exception as e:
            logger.error(f"Query failed: {query}\nError: {e}")
            return {"error": str(e), "query": query}

    # ----------------- User Operations -----------------
    
    def insert_user(self, username: str, email: str, his_job: str, password: str) -> uuid.UUID:
        user_id = uuid.uuid4()
        print(f"📥 Trying to insert user: {username} | {email} | {his_job}")
    
        result = self._execute_cql(
            f"""INSERT INTO {self.KEYSPACE}.users_new 
            (user_id, username, email, his_job, password) 
            VALUES (%s, %s, %s, %s, %s)
            """,
            (user_id, username, email, his_job, password)
        )
    
        print(f"✅ Insert result: {result}")
        return user_id



    def get_user(self, username: str) -> Optional[dict]:
        try:
            query = f"SELECT username, email, password, his_job FROM {self.KEYSPACE}.users_new WHERE username = %s ALLOW FILTERING"
            result = self.session.execute(query, (username,))
            row = result.one()
            if row:
                return {
                    "username": row.username,
                    "email": row.email,
                    "password": row.password,
                    "his_job": row.his_job
                }
                print(f"🔍 Looking for user: {username}")

            return None
        except Exception as e:
            print(f"Error fetching user: {e}")
            return None
        
    def create_user(self, username: str, email: str, password: str, his_job: str) -> bool:
        try:
            query = """
                INSERT INTO users_new (username, email, password, his_job)
                VALUES (%s, %s, %s, %s)
            """
            self.session.execute(query, (username, email, password, his_job))
            return True
        except Exception as e:
            print(f"[❌ Cassandra create_user error]: {e}")
            return False


    def find_user(self, username: str) -> Optional[Dict]:
        """Retrieve user by username"""
        result = self._execute_cql(
            f"SELECT * FROM {self.KEYSPACE}.users_new WHERE username=%s ALLOW FILTERING",
            (username,)
        )
        return result.one() if result else None

    def retrieve_user_id(self, username: str) -> Optional[uuid.UUID]:
        """Get user ID by username"""
        result = self._execute_cql(
            f"SELECT user_id FROM {self.KEYSPACE}.users_new WHERE username=%s ALLOW FILTERING",
            (username,)
        )
        return result.one().user_id if result else None

    # ----------------- Session Operations -----------------
    def create_room_session(self, user_id: Optional[uuid.UUID] = None) -> uuid.UUID:
        """Create a new chat session"""
        session_id = uuid.uuid4()
        self._execute_cql(
            f"INSERT INTO {self.KEYSPACE}.session_table (session_id, user_id) VALUES (%s, %s)",
            (session_id, user_id)
        )
        return session_id

    def get_sessions(self, username: str) -> Dict[str, List[Dict]]:
        """Get all sessions grouped by time period"""
        user_id = self.retrieve_user_id(username)
        if not user_id:
            return {}
            
        sessions = self._execute_cql(
            f"""
            SELECT session_id, title, timestamp 
            FROM {self.KEYSPACE}.session_table 
            WHERE user_id=%s ALLOW FILTERING
            """,
            (user_id,)
        )
        
        if not sessions:
            return {}
            
        grouped = defaultdict(list)
        for session in sessions:
            category = categorize_timestamp(session.timestamp)
            if category:
                grouped[category].append({
                    "session_id": session.session_id,
                    "title": session.title,
                    "timestamp": session.timestamp
                })
        
        # Sort by time categories
        sort_order = ["Today", "Yesterday", "Last 7 Days", "Last 30 Days"]
        return {
            **{k: grouped[k] for k in sort_order if k in grouped},
            **{k: grouped[k] for k in grouped if k not in sort_order}
        }

    # ----------------- Chat Operations -----------------
    def insert_answer(self, session_id: uuid.UUID, username: str, question: str, answer: str) -> uuid.UUID:
        """Store a Q&A pair and update session"""
        user_id = self.retrieve_user_id(username)
        partition_id = uuid.uuid1()
        now = datetime.now()
        
        # Insert response
        self._execute_cql(
            f"""
            INSERT INTO {self.KEYSPACE}.response_table 
            (partition_id, question, answer, timestamp, evaluation)
            VALUES (%s, %s, %s, %s, false)
            """,
            (partition_id, question, answer, now)
        )
        
        # Link response to session
        self._execute_cql(
            f"""
            INSERT INTO {self.KEYSPACE}.response_session 
            (partition_id, session_id)
            VALUES (%s, %s)
            """,
            (partition_id, session_id)
        )
        
        # Update session title if needed
        session = self._execute_cql(
            f"SELECT * FROM {self.KEYSPACE}.session_table WHERE session_id=%s ALLOW FILTERING",
            (session_id,)
        ).one()
        
        if session and session.title is None:
            llm = genai.Client(api_key="AIzaSyAcIGFo53M8vf2eb_UO4JGBYb0an7B8xH4").chats.create(model="gemini-2.0-flash")
            prompt = f"Generate a short chat title based on: '{question}'. Respond with max 3 lowercase words, comma-separated."
            title = llm.send_message(prompt).text
            
            self._execute_cql(
                f"""
                UPDATE {self.KEYSPACE}.session_table 
                SET title=%s, timestamp=%s, user_id=%s 
                WHERE session_id=%s
                """,
                (title, now, user_id, session_id)
            )
        
        return partition_id

    def get_chat_history(self, session_id: str) -> List[Tuple[str, str]]:
        """Retrieve conversation history for a session"""
        session_id = uuid.UUID(session_id)
        response_ids = self._execute_cql(
            f"""
            SELECT partition_id 
            FROM {self.KEYSPACE}.response_session 
            WHERE session_id=%s ALLOW FILTERING
            """,
            (session_id,)
        )
        
        history = []
        for row in response_ids:
            qa_pair = self._execute_cql(
                f"""
                SELECT question, answer 
                FROM {self.KEYSPACE}.response_table 
                WHERE partition_id=%s ALLOW FILTERING
                """,
                (row.partition_id,)
            ).one()
            
            if qa_pair:
                history.append((qa_pair.question, qa_pair.answer))
                
        return history

    # ----------------- Utility Methods -----------------
    def clear_tables(self):
        """Truncate all tables (for development/testing)"""
        tables = [
            "vectores", "users_new", "session_table", 
            "response_session", "response_table"
        ]
        for table in tables:
            self._execute_cql(f"TRUNCATE {self.KEYSPACE}.{table}")

    def retrieve_column_descriptions(self) -> Dict[str, str]:
        """Get schema information for all tables"""
        columns = self._execute_cql(
            f"""
            SELECT table_name, column_name, type 
            FROM system_schema.columns 
            WHERE keyspace_name='{self.KEYSPACE}'
            """
        )
        
        schema = {}
        for row in columns:
            if row.table_name not in schema:
                schema[row.table_name] = ""
            schema[row.table_name] += f"{row.column_name} ({row.type})\n"
            
        return schema

    def execute_statement(self, query: str) -> Any:
        """Raw CQL execution with error handling"""
        return self._execute_cql(query)