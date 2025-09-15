"""
getout_of_text_3: A Python Toolkit for Legal Text Analysis & Open Science
=========================================================================

A comprehensive toolkit designed for legal scholars and researchers working with 
legal corpora, Supreme Court opinions, and English legal datasets.

🎯 Main Features:
  • LegalCorpus: Process and analyze legal document collections
  • Supreme Court data integration and analysis tools
  • Text processing utilities optimized for legal documents
  • Seamless integration with pandas and numpy for data science workflows
  • Support for reproducible research in legal studies

📚 For Legal Scholars & Open Science:
  • Analyze Supreme Court opinions and legal texts
  • Extract insights from legal document databases  
  • Support computational legal research and digital humanities
  • Enable reproducible and transparent legal scholarship

🚀 Quick Start:
    >>> import getout_of_text_3 as got3
    >>> corpus = got3.LegalCorpus()
    >>> # Start analyzing legal texts!

📖 Documentation: https://github.com/atnjqt/getout_of_text3
📦 PyPI: pip install getout-of-text-3

Advancing legal scholarship through open computational tools! ⚖️
"""

from getout_of_text_3._config import options
from getout_of_text_3.corpus import LegalCorpus

def read_corpus(dir_of_text_files=None):
    """
    Convenience function to read COCA corpus files.
    Creates a temporary LegalCorpus instance and loads the data.
    
    Returns the loaded corpus dictionary.
    """
    corpus = LegalCorpus()
    return corpus.read_corpus(dir_of_text_files)

# Expose main functions for easy access
def read_corpora(dir_of_text_files, corpora_name, genre_list=None):
    """
    Convenience function to read COCA corpus files.
    Creates a temporary LegalCorpus instance and loads the data.
    
    Returns the loaded corpus dictionary.
    """
    corpus = LegalCorpus()
    return corpus.read_corpora(dir_of_text_files, corpora_name, genre_list)

def search_keyword_corpus(keyword, db_dict, case_sensitive=False, show_context=True, context_words=5, output='print'):
    """
    Convenience function for keyword search across corpus.
    """
    corpus = LegalCorpus()
    return corpus.search_keyword_corpus(keyword, db_dict, case_sensitive, show_context, context_words, output)

def find_collocates(keyword, db_dict, window_size=5, min_freq=2, case_sensitive=False):
    """
    Convenience function for collocate analysis.
    """
    corpus = LegalCorpus()
    return corpus.find_collocates(keyword, db_dict, window_size, min_freq, case_sensitive)

def keyword_frequency_analysis(keyword, db_dict, case_sensitive=False, relative=True):
    """Convenience function for frequency analysis.

    Parameters mirror LegalCorpus.keyword_frequency_analysis
    """
    corpus = LegalCorpus()
    return corpus.keyword_frequency_analysis(keyword, db_dict, case_sensitive=case_sensitive, relative=relative)

# Backward compatibility / explicit public API
__all__ = [
    'LegalCorpus',
    'read_corpus',
    'read_corpora',
    'search_keyword_corpus',
    'find_collocates',
    'keyword_frequency_analysis',
    'embedding',
    'options',
    '__version__'
]

# Import embedding modules
try:
    from . import legal_bert, embeddinggemma
    # Create embedding namespace
    class EmbeddingModule:
        def __init__(self):
            # Make legal_bert module directly accessible
            setattr(self, 'legal_bert', legal_bert)
            # Make embeddinggemma module directly accessible
            setattr(self, 'gemma', embeddinggemma)
    
    embedding = EmbeddingModule()
except ImportError:
    # If dependencies aren't available, create placeholder
    class LegalBertPlaceholder:
        def pipe(self, *args, **kwargs):
            raise ImportError("Legal-BERT dependencies not installed. Run: pip install transformers torch matplotlib seaborn")
        
        def legal_bert(self, *args, **kwargs):
            raise ImportError("Legal-BERT dependencies not installed. Run: pip install transformers torch matplotlib seaborn")
    
    class EmbeddingGemmaPlaceholder:
        def gemma(self, *args, **kwargs):
            raise ImportError("EmbeddingGemma dependencies not installed. Run: pip install sentence-transformers torch")
        
        def to_json(self, *args, **kwargs):
            raise ImportError("EmbeddingGemma dependencies not installed. Run: pip install sentence-transformers torch")
    
    class EmbeddingModule:
        def __init__(self):
            setattr(self, 'legal_bert', LegalBertPlaceholder())
            setattr(self, 'gemma', EmbeddingGemmaPlaceholder())
    
    embedding = EmbeddingModule()

# make the interactive namespace easier to use
# for `from getout_of_text3 import *` demos.
import getout_of_text_3 as got3
import pandas as pd
import numpy as np
from . import _version
__version__ = _version.get_versions()["version"]