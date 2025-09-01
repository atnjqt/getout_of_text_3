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

# Expose main corpus class
from getout_of_text_3.corpus import LegalCorpus

# Expose main functions for easy access
def read_corpora(dir_of_text_files, corpora_name, genre_list=None):
    """
    Convenience function to read COCA corpus files.
    Creates a temporary LegalCorpus instance and loads the data.
    
    Returns the loaded corpus dictionary.
    """
    corpus = LegalCorpus()
    return corpus.read_corpora(dir_of_text_files, corpora_name, genre_list)

def search_keyword_corpus(keyword, db_dict, case_sensitive=False, show_context=True, context_words=5):
    """
    Convenience function for keyword search across corpus.
    """
    corpus = LegalCorpus()
    return corpus.search_keyword_corpus(keyword, db_dict, case_sensitive, show_context, context_words)

def find_collocates(keyword, db_dict, window_size=5, min_freq=2, case_sensitive=False):
    """
    Convenience function for collocate analysis.
    """
    corpus = LegalCorpus()
    return corpus.find_collocates(keyword, db_dict, window_size, min_freq, case_sensitive)

def keyword_frequency_analysis(keyword, db_dict, case_sensitive=False):
    """
    Convenience function for frequency analysis.
    """
    corpus = LegalCorpus()
    return corpus.keyword_frequency_analysis(keyword, db_dict, case_sensitive)

# ...existing code...
#from getout_of_text_3.io.file import _read_file as read_file
#from getout_of_text_3.tools import sjoin, sjoin_nearest

#import getout_of_text_3.datasets

# make the interactive namespace easier to use
# for `from getout_of_text3 import *` demos.
import getout_of_text_3 as got3
import pandas as pd
import numpy as np

# Move version import to the end to avoid circular import issues
from . import _version
__version__ = _version.get_versions()["version"]