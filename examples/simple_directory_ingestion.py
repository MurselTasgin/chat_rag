# /Users/murseltasgin/projects/chat_rag/examples/simple_directory_ingestion.py
"""
Simple example showing how to ingest documents from a directory
"""
from pipeline import RAGPipeline
from config import Settings


def main():
    """Simple directory ingestion example"""
    
    print("="*80)
    print("SIMPLE DIRECTORY INGESTION EXAMPLE")
    print("="*80)
    
    # Initialize pipeline
    settings = Settings()
    pipeline = RAGPipeline(settings=settings)
    
    # Method 1: Use default directory from settings (DOCUMENTS_INPUT_PATH in .env)
    print("\n📂 Method 1: Using default directory from settings")
    print(f"   Default path: {settings.documents_input_path}")
    print(f"   Recursive: {settings.documents_recursive}")
    
    try:
        # Just call without parameters - uses settings defaults
        results = pipeline.ingest_documents_from_directory()
        
        print(f"\n✓ Ingested {len(results)} documents")
        for file_path, chunks in results.items():
            print(f"  - {file_path}: {len(chunks)} chunks")
    except Exception as e:
        print(f"\n✗ Failed: {e}")
        print(f"\nTip: Create a '{settings.documents_input_path}' directory and add some documents")
    
    # Method 2: Specify custom directory
    print("\n\n📂 Method 2: Using custom directory")
    
    try:
        results = pipeline.ingest_documents_from_directory(
            directory_path="./my_documents",  # Custom path
            recursive=True,
            additional_metadata={"source": "custom_folder"}
        )
        
        print(f"\n✓ Ingested {len(results)} documents from custom folder")
    except Exception as e:
        print(f"\n✗ Failed: {e}")
    
    # Method 3: Filter by file type
    print("\n\n📂 Method 3: Ingest only specific file types")
    
    try:
        # Only PDF files from default directory
        results = pipeline.ingest_documents_from_directory(
            file_pattern="*.pdf",
            additional_metadata={"file_type": "pdf"}
        )
        
        print(f"\n✓ Ingested {len(results)} PDF documents")
    except Exception as e:
        print(f"\n✗ Failed: {e}")
    
    # Query the ingested documents
    print("\n\n🔍 Querying ingested documents")
    
    if pipeline.vector_db.count() > 0:
        results, metadata = pipeline.retrieve(
            query="What documents do we have?",
            top_k=5
        )
        
        print(f"\nFound {len(results)} results:")
        for i, result in enumerate(results, 1):
            print(f"\n[{i}] {result.chunk.doc_title}")
            print(f"    Score: {result.score:.4f}")
            print(f"    Preview: {result.chunk.content[:100]}...")
    else:
        print("\n⚠ No documents in database yet")
    
    print("\n" + "="*80)
    print("Configuration Tips:")
    print("="*80)
    print("""
    1. Set default directory in .env:
       DOCUMENTS_INPUT_PATH=./my_documents
    
    2. Configure recursive scanning:
       DOCUMENTS_RECURSIVE=true
    
    3. Organize your documents:
       ./documents/
       ├── pdfs/
       │   ├── report1.pdf
       │   └── report2.pdf
       ├── docs/
       │   ├── manual.docx
       │   └── guide.md
       └── images/
           └── diagram.png
    
    4. Run ingestion:
       python -c "from pipeline import RAGPipeline; \\
                  RAGPipeline().ingest_documents_from_directory()"
    """)


if __name__ == "__main__":
    main()

