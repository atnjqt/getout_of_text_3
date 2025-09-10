# getout_of_text3: Enhanced Legal Text Analysis Toolkit

The `getout_of_text3` module is a comprehensive Python library for legal scholars and researchers working with **COCA** ([Corpus of Contemporary American English](https://www.english-corpora.org/coca/)) and other legal / natural language text corpora. It provides tools for traditional NLP tasks, embedding models, and integration with AI large language models (LLMs) to facilitate advanced text analysis and support open science research in legal scholarship.

> **Disclaimer:** This project is still in development and may not yet be suitable for production use. The development of this project is heavily reliant on Artificial Intelligence coding tools for staging and deploying this PyPi module. Please use with caution as it is only intended for experimental use cases and explicitly provides no warranty of fitness for any particular task. In no way does this tool provide legal advice, nor do the authors of this module endorse any generative outputs you may observe or experience in using the toolset.

## 🎯 Features for Legal & Linguistic Scholars

The `got3` module aims to provide simpler toolsets to promote the computational forensic linguistic discovery of the 'ordinary meaning' of words in and out of legal contexts using modern techniques, with a focus on delivering an open-source tool built around three main areas of functionality:

1. **Traditional NLP:** corpus analysis, keyword searching, collocate analysis, and frequency studies to support open science research in legal scholarship.
2. **Embedding Models**: tools like Legal-BERT, EmbeddingGemma, and other embedding models for deeper semantic analysis of legal texts using state-of-the-art techniques on devices with limited resources.
3. **AI Language Models**: integration with large language models for advanced text generation and analysis tasks.


### Core Functionality

- **Corpus Linguistics**: Read and manage COCA corpus files across multiple genres
    - **Keyword Search**: Find terms with contextual information across legal texts
    - **Collocate Analysis**: Discover words that frequently appear near target terms
    - **Frequency Analysis**: Analyze term frequency across different legal genres
- **Embedding Models**: Integration with legal-specific BERT models for advanced text analysis
    - **Legal-BERT**: Pre-trained models fine-tuned on legal texts for masked word prediction and semantic analysis
    - **EmbeddingGemma**: Efficient embedding model for general text analysis
- **AI Language Models**: Tools for leveraging AI models in legal text analysis
    - **LLM Integration**: Interfaces for using large language models in legal research
- **Reproducible Research**: Support for open science methodologies with notebooks and structured data outputs


## Installation

You can install `getout_of_text3` using pip:

```bash
pip install getout-of-text-3 -U
```

## Quick Start with Embedding

There are a number of embedding models available for legal text analysis. 
- The recommended model is `nlpaueb/legal-bert-base-uncased`, which is specifically trained on legal documents and is the most popular taged 'legal' on Hugging Face (https://huggingface.co/nlpaueb/legal-bert-base-uncased).

- Other noteable examples include the latest EmbeddingGemma model from Google, `google/embeddinggemma-300m`, which promises to be more efficient and effective across general text analysis (https://huggingface.co/google/embeddinggemma-300m).

### Legal Bert Example

`getout_of_text3` provides a convenient interface to use these models for masked word prediction and other embedding tasks, namely using `got3.embedding.legal_bert.pipe()` function:

```python
### Trying it on got3
import getout_of_text_3 as got3

statement = "Establishing a system for the identification and registration of [MASK] animals and regarding the labelling of beef and beef products."
masked_token="bovine"
token_mask="[MASK]"

results = got3.embedding.legal_bert.pipe(statement=statement, # the input text with a [MASK] token
                                         masked_token=masked_token, # any token
                                         token_mask=token_mask, # Default to [MASK]
                                         top_k=5,  # Set number of top predictions to return
                                         visualize=True, # Set to True to display barchart visualization
                                         json_output=False, # Set to True for JSON output
                                         model_name="nlpaueb/legal-bert-base-uncased") # use small for similar results and lesser footprint
```
```plaintext
Top predictions for masked token (highest to lowest):
1. 'live' - Score: 0.6683
2. 'beef' - Score: 0.1665
3. 'farm' - Score: 0.0316
4. 'pet' - Score: 0.0218
5. 'dairy' - Score: 0.0139)
```
![./img/legal_bert_bovine.png](./img/legal_bert_bovine.png)

### EmbeddingGemma Example

```python
### Trying it on got3
import getout_of_text_3 as got3

#tbd...
```



## Quick Start with COCA Corpus

Supported Genres (COCA):
- Academic (`acad`) - Legal academic texts
- Blog (`blog`) - Legal blogs and commentary  
- Fiction (`fic`) - Legal fiction and narratives
- Magazine (`mag`) - Legal magazine articles
- News (`news`) - Legal news coverage
- Spoken (`spok`) - Legal oral arguments and speeches
- TV/Movie (`tvm`) - Legal drama and media
- Web (`web`) - Legal web content

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

### Complete Research Workflow Example

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

### File Format Support

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

- [GitHub Reference](https://github.com/atnjqt/getout_of_text3) - Full function documentation

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
