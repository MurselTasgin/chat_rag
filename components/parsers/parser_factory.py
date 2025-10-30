# /Users/murseltasgin/projects/chat_rag/components/parsers/parser_factory.py
"""
Parser factory for automatic parser selection
"""
import os
from typing import Optional, List
from .base import BaseParser
from .text_parser import TextParser
from .pdf_parser import PDFParser
from .docx_parser import DOCXParser
from .markdown_parser import MarkdownParser
from .image_parser import ImageParser
from core.exceptions import RAGException


class ParserFactory:
    """Factory for creating and managing document parsers"""
    
    def __init__(self):
        """Initialize parser factory"""
        self._parsers: List[BaseParser] = []
        self._register_default_parsers()
    
    def _register_default_parsers(self):
        """Register default parsers"""
        # Try to register each parser (may fail if dependencies not installed)
        parsers_to_register = [
            (TextParser, {}),
            (MarkdownParser, {}),
            (PDFParser, {'use_unstructured': False}),  # Try PyMuPDF first
            (DOCXParser, {}),
            (ImageParser, {}),
        ]
        
        for parser_class, kwargs in parsers_to_register:
            try:
                parser = parser_class(**kwargs)
                self._parsers.append(parser)
            except RAGException as e:
                # If PyMuPDF fails, try unstructured for PDF
                if parser_class == PDFParser and not kwargs.get('use_unstructured'):
                    try:
                        parser = PDFParser(use_unstructured=True)
                        self._parsers.append(parser)
                        print(f"Info: Using unstructured for PDF parsing (PyMuPDF not available)")
                    except RAGException:
                        print(f"Warning: Could not register {parser_class.__name__}: {e}")
                else:
                    print(f"Warning: Could not register {parser_class.__name__}: {e}")
    
    def register_parser(self, parser: BaseParser):
        """
        Register a custom parser
        
        Args:
            parser: Parser instance to register
        """
        self._parsers.append(parser)
    
    def get_parser(self, file_path: str) -> Optional[BaseParser]:
        """
        Get appropriate parser for a file
        
        Args:
            file_path: Path to the file
        
        Returns:
            Parser instance or None if no parser found
        """
        for parser in self._parsers:
            if parser.supports(file_path):
                return parser
        return None
    
    def parse_file(self, file_path: str, **kwargs) -> str:
        """
        Parse a file using the appropriate parser
        
        Args:
            file_path: Path to the file
            **kwargs: Additional parser parameters
        
        Returns:
            Extracted text content
        
        Raises:
            RAGException: If no parser found or parsing fails
        """
        if not os.path.exists(file_path):
            raise RAGException(f"File not found: {file_path}")
        
        parser = self.get_parser(file_path)
        if parser is None:
            ext = os.path.splitext(file_path)[1]
            raise RAGException(
                f"No parser available for file type: {ext}\n"
                f"Supported parsers: {[p.get_name() for p in self._parsers]}"
            )
        
        return parser.parse(file_path, **kwargs)
    
    def get_metadata(self, file_path: str, **kwargs) -> dict:
        """
        Get metadata from a file
        
        Args:
            file_path: Path to the file
            **kwargs: Additional parser parameters
        
        Returns:
            Metadata dictionary
        """
        parser = self.get_parser(file_path)
        if parser is None:
            return {'file_name': os.path.basename(file_path)}
        
        return parser.get_metadata(file_path, **kwargs)
    
    def list_parsers(self) -> List[str]:
        """
        List all registered parsers
        
        Returns:
            List of parser names
        """
        return [parser.get_name() for parser in self._parsers]
    
    def list_supported_extensions(self) -> List[str]:
        """
        List all supported file extensions
        
        Returns:
            List of file extensions
        """
        extensions = set()
        
        # Check each parser with common extensions
        test_extensions = [
            '.txt', '.pdf', '.docx', '.md', '.png', '.jpg', '.jpeg',
            '.gif', '.bmp', '.tiff', '.html', '.xml', '.json', '.csv'
        ]
        
        for ext in test_extensions:
            test_file = f"test{ext}"
            if self.get_parser(test_file) is not None:
                extensions.add(ext)
        
        return sorted(list(extensions))

