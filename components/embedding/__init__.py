# /Users/murseltasgin/projects/chat_rag/components/embedding/__init__.py
"""
Embedding component
"""
from .base import BaseEmbedding
from .sentence_transformer_embedding import SentenceTransformerEmbedding

__all__ = ['BaseEmbedding', 'SentenceTransformerEmbedding']

