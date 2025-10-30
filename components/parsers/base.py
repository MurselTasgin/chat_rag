# /Users/murseltasgin/projects/chat_rag/components/parsers/base.py
"""
Base parser abstraction
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseParser(ABC):
    """Base class for document parsers"""
    
    @abstractmethod
    def parse(self, file_path: str, **kwargs) -> str:
        """
        Parse a document and return its text content
        
        Args:
            file_path: Path to the document
            **kwargs: Additional parser-specific parameters
        
        Returns:
            Extracted text content
        """
        pass
    
    @abstractmethod
    def supports(self, file_path: str) -> bool:
        """
        Check if this parser supports the given file
        
        Args:
            file_path: Path to the file
        
        Returns:
            True if file is supported, False otherwise
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the parser name"""
        pass
    
    def get_metadata(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """
        Extract metadata from the document (optional)
        
        Args:
            file_path: Path to the document
            **kwargs: Additional parameters
        
        Returns:
            Dictionary of metadata
        """
        import os
        return {
            'file_name': os.path.basename(file_path),
            'file_size': os.path.getsize(file_path),
            'parser': self.get_name()
        }

