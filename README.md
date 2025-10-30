# /Users/murseltasgin/projects/chat_rag/README.md
# Advanced RAG System with Conversational Context

A modular, production-ready Retrieval-Augmented Generation (RAG) system with sophisticated ingestion and query pipelines.

## Features

- **Semantic Chunking**: Context-preserving document chunking with overlap
- **Contextual RAG**: Document metadata enrichment and context awareness
- **Query Understanding**: Automatic query clarification and expansion
- **Hybrid Retrieval**: Combines vector search (semantic) with BM25 (keyword)
- **Intelligent Reranking**: LLM-based or Cross-Encoder reranking for optimal results
- **Conversation Tracking**: Multi-turn conversation support with reference resolution
- **Smart Search Strategy**: Automatically selects optimal retrieval method
- **Modular Architecture**: Pluggable components following SOLID principles

## Architecture

```
chat_rag/
├── config/                  # Configuration management
│   ├── __init__.py
│   └── settings.py         # Central config from .env
├── core/                   # Core data models and exceptions
│   ├── __init__.py
│   ├── models.py          # Data models (DocumentChunk, RetrievalResult, etc.)
│   └── exceptions.py      # Custom exceptions
├── components/            # Pluggable components
│   ├── chunker/          # Text chunking strategies
│   │   ├── base.py       # Base chunker interface
│   │   └── semantic_chunker.py
│   ├── embedding/        # Embedding models
│   │   ├── base.py       # Base embedding interface
│   │   └── sentence_transformer_embedding.py
│   ├── vectordb/         # Vector database providers
│   │   ├── base.py       # Base vectordb interface
│   │   └── chroma_vectordb.py
│   ├── llm/              # LLM providers
│   │   ├── base.py       # Base LLM interface
│   │   └── azure_openai_llm.py
│   ├── retriever/        # Retrieval strategies
│   │   ├── base.py       # Base retriever interface
│   │   └── hybrid_retriever.py
│   ├── query_processor/  # Query understanding and enhancement
│   │   └── query_enhancer.py
│   ├── reranker/         # Result reranking
│   │   └── reranker.py
│   ├── conversation/     # Conversation management
│   │   └── conversation_manager.py
│   ├── document_processor/ # Document preprocessing
│   │   └── document_processor.py
│   └── contextual_enhancer/ # Contextual enrichment
│       └── contextual_enhancer.py
├── pipeline/             # Main orchestration
│   ├── __init__.py
│   └── rag_pipeline.py   # Main RAG pipeline
├── main_new.py           # Example usage
├── requirements.txt      # Dependencies
└── .env.example         # Environment variables template
```

## Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd chat_rag
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download NLTK data**
```bash
python setup_nltk.py
```

4. **Configure environment**
```bash
cp env.example .env
# Edit .env with your Azure OpenAI credentials or set LLM_PROVIDER=ollama
```

## Configuration

All configuration is centralized in `config/settings.py` and reads from `.env` file:

```env
# Azure OpenAI
AZURE_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_API_KEY=your-api-key
AZURE_DEPLOYMENT=gpt-4o

# Embedding Model
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Vector Database
VECTOR_DB_PATH=./chroma_db
VECTOR_DB_COLLECTION=documents

# Chunking
CHUNK_SIZE=512
CHUNK_OVERLAP=128
MIN_CHUNK_SIZE=100

# Retrieval
DEFAULT_TOP_K=5
VECTOR_WEIGHT=0.7
BM25_WEIGHT=0.3

# Conversation
ENABLE_CONVERSATION=true
MAX_CONVERSATION_HISTORY=10

# Reranker Configuration
# Options: 'llm' or 'cross_encoder'
RERANKER_TYPE=cross_encoder
CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# Document Parsing
PDF_PARSER_BACKEND=pymupdf
OCR_LANGUAGE=eng
DOCUMENTS_INPUT_PATH=./documents
DOCUMENTS_RECURSIVE=true
```

## Usage

### CLI Chat Application

Run the command-line interface for conversational chat:

```bash
python main_new.py
```

This will:
1. Automatically ingest documents from the configured input folder (if not already ingested)
2. Track ingested documents to avoid re-processing
3. Start an interactive chat session

### Web Application

Run the Flask web application for a modern browser-based chat interface:

```bash
python app.py
```

Then open your browser to: `http://localhost:5000`

Features:
- 🎨 Beautiful, modern UI with gradient design
- 💬 Real-time conversational chat
- 📚 Source citations with relevance scores
- 📊 Document statistics
- 🗑️ Clear conversation history
- 📱 Responsive design

### Basic Usage

```python
from pipeline import RAGPipeline
from config import Settings

# Initialize
settings = Settings()
rag_pipeline = RAGPipeline(settings=settings)

# Ingest documents from default directory (set in .env: DOCUMENTS_INPUT_PATH)
results = rag_pipeline.ingest_documents_from_directory()

# Or specify a custom directory
results = rag_pipeline.ingest_documents_from_directory(
    directory_path="./my_documents",
    recursive=True,
    file_pattern="*.pdf"  # Optional: filter by file type
)

# Ingest a single document
chunks = rag_pipeline.ingest_document_from_file("./documents/report.pdf")

# Retrieve relevant information
results, metadata = rag_pipeline.retrieve(
    query="What is John Smith's role?",
    top_k=5
)

# Format results for LLM
context = rag_pipeline.get_retrieval_context(results)
print(context)
```

### Conversational Usage

```python
# Turn 1
results1, _ = rag_pipeline.retrieve("What is John Smith's role?")
assistant_response = "John Smith is a Senior Software Engineer."
rag_pipeline.add_assistant_response(assistant_response)

# Turn 2 - Ambiguous reference resolved automatically
results2, _ = rag_pipeline.retrieve("How old is he?")
# System automatically clarifies to "How old is John Smith?"

# View conversation history
print(rag_pipeline.get_conversation_summary())
```

### Custom Components

You can replace any component with your own implementation:

```python
from components.llm import BaseLLM
from components.embedding import BaseEmbedding
from components.reranker import CrossEncoderReranker

# Use cross-encoder reranker for better performance
cross_encoder = CrossEncoderReranker(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
)
rag_pipeline = RAGPipeline(reranker=cross_encoder, settings=settings)

# Custom LLM
class MyCustomLLM(BaseLLM):
    def generate(self, messages, temperature=0.3, max_tokens=200, **kwargs):
        # Your implementation
        pass
    
    def get_name(self):
        return "MyCustomLLM"
    
    def get_model_name(self):
        return "custom-model"

# Use custom component
custom_llm = MyCustomLLM()
rag_pipeline = RAGPipeline(llm_model=custom_llm, settings=settings)
```

## Adding New Components

### Adding a New LLM Provider

1. Create a new file: `components/llm/my_llm.py`
2. Implement `BaseLLM` interface
3. Import in `components/llm/__init__.py`
4. Use in pipeline initialization

```python
from components.llm.base import BaseLLM

class MyLLM(BaseLLM):
    def generate(self, messages, temperature=0.3, max_tokens=200, **kwargs):
        # Implementation
        pass
    
    def get_name(self):
        return "MyLLM"
    
    def get_model_name(self):
        return "my-model-v1"
```

### Adding a New Vector Database

1. Create: `components/vectordb/my_vectordb.py`
2. Implement `BaseVectorDB` interface
3. Import in `components/vectordb/__init__.py`

## Design Principles

This codebase follows these key principles:

1. **Abstraction**: No hardcoded technology-specific code in main components
2. **Separation of Concerns**: Data access, business logic, and presentation are separated
3. **Strategy Pattern**: Algorithms are pluggable (retrieval, chunking, etc.)
4. **Interface Segregation**: Components depend only on methods they use
5. **Modularity**: Code is organized in small, focused modules (< 500 lines per file)
6. **Centralized Configuration**: All config through central settings module
7. **Resilience**: Exception handling and graceful degradation
8. **Extensibility**: Easy to add new components without changing existing code

## Testing

```bash
# Run unit tests
python -m pytest tests/

# Run integration tests
python -m pytest tests/integration/

# Run end-to-end tests
python -m pytest tests/e2e/

# Run example with cross-encoder reranker
python examples/cross_encoder_reranker_example.py
```

## Reranking Strategies

The system supports two reranking strategies:

1. **LLM Reranker**: Uses language model for relevance assessment (flexible but slower)
2. **Cross-Encoder Reranker**: Uses specialized cross-encoder model (fast and accurate)

See [Reranker Guide](docs/RERANKER_GUIDE.md) for detailed comparison and usage.

**Quick Start with Cross-Encoder:**
```python
# Set in .env
RERANKER_TYPE=cross_encoder

# Or in code
from components.reranker import CrossEncoderReranker
reranker = CrossEncoderReranker()
pipeline = RAGPipeline(reranker=reranker)
```

## Logging and Metrics

The system includes:
- Standard logging for debugging
- Token usage tracking
- Performance metrics
- Input/output logging for LLM calls

## Health Checks

For API deployments, health check endpoints verify:
- LLM connectivity
- Vector database status
- Embedding model availability

## Contributing

1. Follow the existing code structure
2. Keep files under 500-600 lines
3. Keep functions/methods under 20-30 lines
4. Add unit tests for new components
5. Update documentation

## License

[Your License]

## Support

For issues and questions, please open a GitHub issue.

