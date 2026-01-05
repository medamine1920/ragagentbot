## 🚀 RAGAgentBot – Production-Ready Retrieval-Augmented Generation System

RAGAgentBot is a full-stack Retrieval-Augmented Generation (RAG) application designed for real-world decision-support use cases. The system enables users to upload documents, retrieve relevant context using vector search, and generate grounded answers with a Large Language Model (LLM).

Unlike demo-level RAG projects, RAGAgentBot focuses on scalability, performance, and cost efficiency through semantic caching, persistent storage, and containerized deployment.

### Key Capabilities
- 📄 Document ingestion and chunking (PDF support)
- 🔍 Semantic search using FAISS + HuggingFace embeddings
- 🧠 Context-aware LLM responses powered by Google Gemini
- ⚡ Semantic caching to reduce redundant LLM calls
- 💬 Persistent chat history per user and session
- 🔐 Authentication and role-based access control (admin/user)
- 🗄️ Cassandra / Astra DB for vector and metadata storage
- 🐳 Fully containerized using Docker and Docker Compose

### Designed For
- Legal and regulatory decision support
- Knowledge-based chatbots
- Enterprise document Q&A systems
- Cost-efficient LLM applications

This project demonstrates end-to-end system design, backend engineering, vector search, and applied NLP in a production-oriented architecture.
