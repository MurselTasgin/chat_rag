# /Users/murseltasgin/projects/chat_rag/components/embedding/sentence_transformer_embedding.py
"""
Sentence Transformer embedding implementation
"""
from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from .base import BaseEmbedding
from core.exceptions import EmbeddingException


class SentenceTransformerEmbedding(BaseEmbedding):
    """Sentence Transformer embedding implementation"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize Sentence Transformer model
        
        Args:
            model_name: Name of the model to use
        """
        self.model_name = model_name
        
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            raise EmbeddingException(f"Failed to load model {model_name}: {e}")
    
    def encode(
        self,
        texts: Union[str, List[str]],
        convert_to_tensor: bool = False,
        **kwargs
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Encode text(s) to embedding(s)"""
        try:
            embeddings = self.model.encode(
                texts,
                convert_to_tensor=convert_to_tensor,
                **kwargs
            )
            return embeddings
        except Exception as e:
            raise EmbeddingException(f"Encoding failed: {e}")
    
    def get_name(self) -> str:
        """Get the embedding model name"""
        return f"SentenceTransformer-{self.model_name}"
    
    def get_dimension(self) -> int:
        """Get the embedding dimension"""
        return self.model.get_sentence_embedding_dimension()

