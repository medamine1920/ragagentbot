import asyncio
from pathlib import Path
from rag_agent import RAGAgent

async def main():
    # Initialize the RAG agent
    agent = RAGAgent()
    
    try:
        # Test document processing
        print("Testing document processing...")
        test_file_path = "ragagent\\backend\\APP\\test_documents\\Motivation_letter_VML.pdf"  # Change this to your test file
        if Path(test_file_path).exists():
            with open(test_file_path, "rb") as f:
                file_bytes = f.read()
            success = await agent.process_document(file_bytes, test_file_path)
            print(f"Document processing {'succeeded' if success else 'failed'}")
        else:
            print(f"Test file {test_file_path} not found - skipping document processing test")
        
        # Test query functionality
        print("\nTesting query functionality...")
        test_queries = [
            "What is this document about?",
            "Summarize the key points",
            "Explain the main concepts"
        ]
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            response = await agent.generate_response(query)
            print(f"Answer: {response['answer']}")
            print(f"Confidence: {response['confidence']}")
            if response['sources']:
                print(f"Sources: {response['sources']}")
            print(f"From cache: {response.get('from_cache', False)}")
        
    finally:
        # Clean up
        agent.close()

if __name__ == "__main__":
    asyncio.run(main())