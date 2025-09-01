# getout_of_text3: Enhanced Legal Text Analysis Toolkit

getout_of_text3 is a comprehensive Python library for legal scholars and researchers working with COCA (Corpus of Contemporary American English) and other legal text corpora. It provides advanced tools for corpus analysis, keyword searching, collocate analysis, and frequency studies to support open science research in legal scholarship.

> **AI Disclaimer** This project is still in development and may not yet be suitable for production use. The development of this project is heavily reliant on AI CoPilot tools for staging and creating this pypi module. Please use with caution as it's only intended for experimental use cases and provides no warranty of fitness for any particular task.

## 🎯 Features for Legal Scholars

### Core Functionality
- **Corpus Loading**: Read and manage COCA corpus files across multiple genres
- **Keyword Search**: Find terms with contextual information across legal texts
- **Collocate Analysis**: Discover words that frequently appear near target terms
- **Frequency Analysis**: Analyze term frequency across different legal genres
- **Reproducible Research**: Support for open science methodologies

### Supported Genres
- Academic (`acad`) - Legal academic texts
- Blog (`blog`) - Legal blogs and commentary  
- Fiction (`fic`) - Legal fiction and narratives
- Magazine (`mag`) - Legal magazine articles
- News (`news`) - Legal news coverage
- Spoken (`spok`) - Legal oral arguments and speeches
- TV/Movie (`tvm`) - Legal drama and media
- Web (`web`) - Legal web content

## Installation

You can install getout_of_text3 using pip:

```bash
pip install getout-of-text-3
```

## Quick Start

### Method 1: Using Convenience Functions (Recommended for Beginners)

```python
import getout_of_text_3 as got3

# 1. Read COCA corpus files
corpus_data = got3.read_corpora("path/to/coca/files", "my_legal_corpus")

# 2. Search for legal terms with context
results = got3.search_keyword_corpus(
    keyword="constitutional",
    db_dict=corpus_data,
    case_sensitive=False,
    show_context=True,
    context_words=5
)

# 3. Find collocates (words that appear near your target term)
collocates = got3.find_collocates(
    keyword="justice",
    db_dict=corpus_data,
    window_size=5,
    min_freq=2
)

# 4. Analyze frequency across genres
freq_analysis = got3.keyword_frequency_analysis(
    keyword="legal",
    db_dict=corpus_data
)
```

### Method 2: Using LegalCorpus Class (Object-Oriented Approach)

```python
import getout_of_text_3 as got3

# Initialize the corpus manager
corpus = got3.LegalCorpus()

# Load multiple corpora
constitutional_corpus = corpus.read_corpora("constitutional-texts", "constitutional")
case_law_corpus = corpus.read_corpora("case-law-texts", "cases")

# Manage your corpus collection
print("Available corpora:", corpus.list_corpora())
corpus.corpus_summary()

# Access specific corpus for analysis
constitutional_data = corpus.get_corpus("constitutional")

# Perform analyses using class methods
search_results = corpus.search_keyword_corpus("amendment", constitutional_data)
collocate_results = corpus.find_collocates("amendment", constitutional_data)
freq_results = corpus.keyword_frequency_analysis("amendment", constitutional_data)
```

## Complete Research Workflow Example

Here's a complete example for analyzing constitutional language across COCA genres:

```python
import getout_of_text_3 as got3

# Step 1: Load your COCA corpus data
print("Loading COCA corpus for constitutional analysis...")
corpus_data = got3.read_corpora("coca-samples-text", "constitutional_study")

# Step 2: Search for constitutional terms with context
print("Searching for 'constitutional' with context...")
constitutional_results = got3.search_keyword_corpus(
    "constitutional", 
    corpus_data, 
    show_context=True, 
    context_words=4
)

# Step 3: Find collocates to understand language patterns
print("Finding collocates for 'constitutional'...")
constitutional_collocates = got3.find_collocates(
    "constitutional", 
    corpus_data, 
    window_size=4, 
    min_freq=2
)

# Step 4: Analyze frequency patterns across genres
print("Analyzing frequency patterns...")
constitutional_freq = got3.keyword_frequency_analysis(
    "constitutional", 
    corpus_data
)

print("🎯 Constitutional Language Analysis Complete!")
print("Results available for further statistical analysis and publication.")
```

## File Format Support

The toolkit supports COCA corpus files in these formats:
- `text_<genre>.txt` - Standard COCA text files
- `db_<genre>.txt` - COCA database files  
- Tab-separated values with text content
- Custom CSV/TSV formats with pandas integration

## Advanced Features

### Text Processing
- NLTK integration for advanced tokenization (with fallback methods)
- Case-sensitive and case-insensitive search options
- Flexible window sizes for collocate analysis
- Customizable frequency thresholds

### Research Support
- Structured data outputs for statistical analysis
- Reproducible methodology documentation
- Integration with pandas for data science workflows
- Support for multiple corpus comparison studies

## Dependencies

- `pandas >= 1.0` - Data manipulation and analysis
- `numpy >= 1.18` - Numerical computing
- `nltk >= 3.8` - Natural language processing (optional but recommended)

NLTK provides enhanced tokenization but the toolkit will use fallback methods if not available.

## For Legal Researchers

This toolkit is specifically designed to support:

- **Constitutional Law Research** - Analyze constitutional language patterns across genres
- **Judicial Opinion Analysis** - Study linguistic patterns in legal decisions  
- **Legal Corpus Linguistics** - Examine legal language evolution over time
- **Comparative Legal Analysis** - Compare legal language usage across different contexts
- **Open Science Initiatives** - Enable reproducible legal research methodologies
- **Digital Humanities** - Support computational approaches to legal scholarship

## Documentation

- [Complete Analysis Guide](COCA_ANALYSIS_GUIDE.md) - Detailed usage examples
- [API Reference](https://github.com/atnjqt/getout_of_text3) - Full function documentation
- [Research Examples](demo_enhanced_functionality.py) - Sample research workflows

## Contributing

We welcome contributions from legal scholars and developers! Please see our contributing guidelines and feel free to submit issues or pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this toolkit in your research, please cite:

```
Jacquot, E. (2025). getout_of_text3: A Python Toolkit for Legal Text Analysis and Open Science. 
GitHub. https://github.com/atnjqt/getout_of_text3
```

## Support

For questions, issues, or feature requests, please visit our [GitHub repository](https://github.com/atnjqt/getout_of_text3) or contact the development team.

---

**Advancing legal scholarship through open computational tools! ⚖️**
