import os
import json
from typing import Dict, List, Optional
from dotenv import load_dotenv
import google.generativeai as genai
from services.cassandra_connector import CassandraConnector
from services.TEXT2CQL_PROMPT import PROMPT_FIX_CQL_V2

load_dotenv()

class CQLAgent:
    def __init__(self, keyspace: str = None):
        self.connector = CassandraConnector()
        self.keyspace = keyspace or os.getenv("CASSANDRA_KEYSPACE", "rag_keyspace")
        self.schema = self._load_schema()
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel('gemini-2.0-flash')

    def _load_schema(self) -> Dict:
        """Load and format schema information"""
        schema = self.connector.get_keyspace_schema()
        
        formatted = {
            "tables": {},
            "partition_keys": {},
            "clustering_columns": {}
        }
        
        for table_name, metadata in schema.items():
            formatted["tables"][table_name] = {
                "columns": [
                    f"{col['column_name']} ({col['type']})" 
                    for col in metadata["columns"]
                ],
                "description": f"Table {table_name} in keyspace {self.keyspace}"
            }
            
            if metadata["partition_keys"]:
                formatted["partition_keys"][table_name] = metadata["partition_keys"]
                
            if metadata["clustering_columns"]:
                formatted["clustering_columns"][table_name] = metadata["clustering_columns"]
        
        return formatted

    def generate_cql(self, natural_language: str) -> Dict:
        """Convert natural language to CQL"""
        prompt = self._build_generation_prompt(natural_language)
        response = self._call_llm(prompt)
        return self._parse_response(response)

    def execute_query_chain(self, queries: List[Dict]) -> Dict:
        """Execute a sequence of dependent queries"""
        results = {}
        context = {}
        
        for query_obj in queries:
            query = query_obj["query"]
            
            if query_obj.get("needs_context", False):
                query = self._adapt_query(query, context)
            
            result = self._execute_query(query)
            
            if "error" in result:
                return {
                    "error": result["error"],
                    "failed_query": query,
                    "successful_results": results
                }
            
            context[query] = result
            results[query] = {
                "data": result,
                "metadata": query_obj
            }
        
        return {"results": results}

    def _build_generation_prompt(self, question: str) -> str:
        """Construct prompt for CQL generation"""
        return f"""
        Convert this natural language question to Cassandra CQL:
        Question: {question}
        
        Schema:
        {json.dumps(self.schema['tables'], indent=2)}
        
        Partition Keys:
        {json.dumps(self.schema['partition_keys'], indent=2)}
        
        Clustering Columns:
        {json.dumps(self.schema['clustering_columns'], indent=2)}
        
        Rules:
        1. No JOINs or subqueries
        2. Use ALLOW FILTERING only when necessary
        3. Return JSON format with list of queries
        4. Mark queries needing context with 'needs_context'
        
        Output format:
        {{
            "queries": [
                {{
                    "query": "SELECT ...",
                    "explanation": "...",
                    "needs_context": boolean
                }}
            ]
        }}
        """

    def _call_llm(self, prompt: str) -> str:
        """Call Gemini LLM"""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error calling LLM: {str(e)}"

    def _parse_response(self, response: str) -> Dict:
        """Parse LLM response"""
        try:
            clean = response.strip().replace("```json", "").replace("```", "")
            return json.loads(clean)
        except json.JSONDecodeError:
            return {"error": "Invalid response format", "raw_response": response}

    def _adapt_query(self, query: str, context: Dict) -> str:
        """Adapt query using previous results"""
        prompt = f"""
        Adapt this CQL query using available context:
        Original Query: {query}
        
        Context:
        {json.dumps(context, indent=2)}
        
        Return ONLY the modified CQL query.
        """
        return self._call_llm(prompt).strip()

    def _execute_query(self, query: str) -> List[Dict]:
        """Execute with automatic error correction"""
        result = self.connector.execute_query(query)
        
        if not isinstance(result, dict) or "error" not in result:
            return result
            
        # Attempt to fix the error
        fixed_query = self._fix_cql_error(query, result["error"])
        return self.connector.execute_query(fixed_query)

    def _fix_cql_error(self, query: str, error: str) -> str:
        """Correct CQL errors using LLM"""
        prompt = PROMPT_FIX_CQL_V2.format(
            error_message=error,
            schema=json.dumps(self.schema["tables"], indent=2),
            partition_keys=json.dumps(self.schema["partition_keys"], indent=2),
            clustering_keys=json.dumps(self.schema["clustering_columns"], indent=2),
            statement=query
        )
        
        response = self._call_llm(prompt)
        return response.strip().replace("```cql", "").replace("```", "")

    def close(self):
        """Clean up resources"""
        self.connector.close()