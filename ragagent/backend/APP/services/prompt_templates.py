from langchain.prompts import PromptTemplate

class RAGPrompts:
    @staticmethod
    def get_default_qa_prompt():
        """Default QA prompt for RAG system"""
        return PromptTemplate(
            template="""Use the following pieces of context to answer the question at the end. 
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            When answering, try to be concise but thorough.

            Context:
            {context}

            Question: {question}
            Answer:""",
            input_variables=["context", "question"]
        )
    
    @staticmethod
    def get_query_transform_prompt():
        """Prompt for transforming natural language to search query"""
        return PromptTemplate(
            template="""Convert the following question into an optimized search query for database retrieval.
            Consider synonyms and alternative phrasings that might appear in the database.

            Original Question: {question}
            Optimized Search Query:""",
            input_variables=["question"]
        )
    
    @staticmethod
    def get_summarization_prompt():
        """Prompt for summarizing retrieved documents"""
        return PromptTemplate(
            template="""Summarize the following document content in 2-3 sentences, 
            preserving key information that might be relevant to the user's question.

            Document Content:
            {document_content}

            Concise Summary:""",
            input_variables=["document_content"]
        )