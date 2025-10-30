# /Users/murseltasgin/projects/chat_rag/examples/ollama_example.py
"""
Example demonstrating Ollama LLM usage
"""
from pipeline import RAGPipeline
from config import Settings
from components.llm import OllamaLLM


def main():
    """Demonstrate Ollama LLM usage"""
    
    print("="*80)
    print("OLLAMA LLM EXAMPLE")
    print("="*80)
    
    # Method 1: Use Ollama through settings
    print("\n📋 Method 1: Configure via .env")
    print("""
    Set in .env:
    LLM_PROVIDER=ollama
    OLLAMA_MODEL=llama2
    OLLAMA_BASE_URL=http://localhost:11434
    """)
    
    # Method 2: Create Ollama LLM directly
    print("\n📋 Method 2: Create OllamaLLM directly")
    
    try:
        # Initialize Ollama LLM
        ollama_llm = OllamaLLM(
            model="llama2",  # or "mistral", "codellama", etc.
            base_url="http://localhost:11434",
            timeout=120
        )
        
        print(f"✓ Connected to Ollama")
        print(f"  Model: {ollama_llm.get_model_name()}")
        print(f"  Provider: {ollama_llm.get_name()}")
        
        # List available models
        models = ollama_llm.list_available_models()
        if models:
            print(f"\n📦 Available models:")
            for model in models:
                print(f"  - {model}")
        
        # Test generation
        print("\n🧪 Testing text generation...")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain what machine learning is in one sentence."}
        ]
        
        response = ollama_llm.generate(messages, temperature=0.7, max_tokens=100)
        print(f"\nResponse: {response}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("  1. Install Ollama: https://ollama.ai/")
        print("  2. Start Ollama server: ollama serve")
        print("  3. Pull a model: ollama pull llama2")
        return
    
    # Method 3: Use Ollama in RAG pipeline
    print("\n\n" + "="*80)
    print("USING OLLAMA IN RAG PIPELINE")
    print("="*80)
    
    try:
        # Create pipeline with Ollama
        settings = Settings()
        settings.llm_provider = "ollama"
        settings.ollama_model = "llama2"
        
        pipeline = RAGPipeline(
            llm_model=ollama_llm,  # Inject our Ollama LLM
            settings=settings
        )
        
        print(f"\n✓ RAG Pipeline initialized with Ollama")
        print(f"  LLM: {pipeline.llm_model.get_name()} - {pipeline.llm_model.get_model_name()}")
        
        # Test with sample document
        sample_doc = """
        Machine Learning Overview
        
        Machine learning is a subset of artificial intelligence that enables systems to learn
        from data without being explicitly programmed. It has three main types:
        
        1. Supervised Learning: Learning from labeled data
        2. Unsupervised Learning: Finding patterns in unlabeled data
        3. Reinforcement Learning: Learning through trial and error
        
        Popular algorithms include neural networks, decision trees, and support vector machines.
        """
        
        print("\n📥 Ingesting sample document...")
        chunks = pipeline.ingest_document(
            document_text=sample_doc,
            doc_id="ml_overview",
            doc_title="Machine Learning Overview"
        )
        print(f"✓ Created {len(chunks)} chunks")
        
        print("\n🔍 Testing retrieval...")
        results, metadata = pipeline.retrieve(
            query="What is supervised learning?",
            top_k=2
        )
        
        print(f"\n✓ Retrieved {len(results)} results")
        for i, result in enumerate(results, 1):
            print(f"\n[Result {i}]")
            print(f"Score: {result.score:.4f}")
            print(f"Content: {result.chunk.content[:150]}...")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
    
    # Performance comparison
    print("\n\n" + "="*80)
    print("OLLAMA vs AZURE OPENAI COMPARISON")
    print("="*80)
    print("""
    Ollama (Local):
      ✓ Free to use
      ✓ No API costs
      ✓ Privacy (data stays local)
      ✓ No rate limits
      ✓ Offline capable
      - Requires local GPU/CPU
      - Slower than cloud APIs
      - Limited to available models
      - Requires ~8GB+ RAM
    
    Azure OpenAI (Cloud):
      ✓ Fast inference
      ✓ State-of-the-art models (GPT-4)
      ✓ No local resources needed
      ✓ Consistent performance
      - Costs per token
      - Requires internet
      - Rate limited
      - Data sent to cloud
    
    Recommended Models for Ollama:
      - llama2 (7B): Good general purpose, fast
      - mistral (7B): Better reasoning, recommended
      - codellama (7B): Good for code
      - mixtral (8x7B): Best quality, slower
      - phi (2.7B): Fastest, good for simple tasks
    """)


if __name__ == "__main__":
    main()

