# getout_of_text3: Enhanced Legal Text Analysis Toolkit

The `getout_of_text3` module is a comprehensive Python library promoting open and reproducible computational forensic linguistics toolsets for data scientists and legal scholars performing textual analysis with popular corpora such as **COCA** ([Corpus of Contemporary American English](https://www.english-corpora.org/coca/)), **SCOTUS** [Library of Congress US Report Collection](https://www.loc.gov/collections/united-states-reports/), and other legal / natural language text corpora.

## 🎯 Overview of Features for Legal & Linguistic Scholars

The `got3` module aims to provide simpler toolsets to promote the discovery of the *'ordinary meaning'* of words in and out of legal contexts using computational techniques, with a focus on delivering an open-source tool built around three main areas of functionality:

- 📚 **Corpus Linguistics**: Read and manage COCA corpus files across multiple genres
    - 🕵 **Keyword Search**: Find terms with contextual information across legal texts
    - 🔍 **Collocate Analysis**: Discover words that frequently appear near target terms
    - 📊 **Frequency Analysis**: Analyze term frequency across different legal genres
- 🤗 **Embedding Models**: Integration with legal-specific BERT models for advanced text analysis
    - **`Legal-BERT`**: Pre-trained models fine-tuned on legal texts for masked word prediction and semantic analysis
    - **`EmbeddingGemma`**: Efficient embedding model for general text analysis
- 🤖 **AI Language Models**: Tools for leveraging AI models in legal text analysis
    - **LLM Integration**: Interfaces for using large language models in legal research
- 🔬 **Reproducible Research**: Support for open science methodologies with notebooks and structured data outputs
    - 🧑‍💻 **Demonstration Notebooks**: Jupyter notebooks showcasing various use cases and methodologies for how to use the tool, with limited compute and/or cloud resources.
    - 📈 **Data Outputs**: Structured outputs suitable for statistical analysis and publication


## Getting Started

### Installation

You can install `getout_of_text3` using pip. I recommend setting up a virtual environment using [venv](https://docs.python.org/3/library/venv.html) or [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) to manage dependencies.

```bash
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate
pip install getout-of-text-3 -U
```

The below examples demonstrate how to use the `getout_of_text3` module for various tasks using corpus linguistics tools, embedding models, and AI language models.

______________________

### Corpus of Contemporary American English (COCA)

If you have access and paid for the COCA corpus (https://www.corpusdata.org/purchase.asp with academic & commercial licenses), you can use the `got3` module to read and analyze the corpus files. Please ensure you comply with the licensing terms of COCA when using the corpus data.

> 📝 Note: The COCA corpus is a large and diverse corpus of American English, and it DOES contain sensitive or proprietary information. Please use the corpus responsibly and in accordance with the licensing terms. English Corpora scrubs 95%/5% with 10 `@` signs, so you may notice that in search results as an effort to promote fair use doctrine in copyright law. The database maintainers also add a watermark throughtout the text content that deviates from the real content, and periodically scan the public web for distribution of this content.

#### Read the Dataset

- the `./coca-text/` directory should contain the COCA text files you downloaded from the English Corpora website, such as `text_acad.txt`, `text_blog.txt`, etc. It's organized by genre and year, except for Web & Blog that are by index.

```python
### Trying it on got3
import getout_of_text_3 as got3

coca_corpus = got3.read_corpus('./coca-text/')
```

#### Search the Corpus for a Keyword

- your `coca_corpus` is a **dictionary of dataframes**, one for each genre, that you can use for further analysis.

```python 
# use time elapse to show query times. multiprocessing is available for faster searches.
import pandas as pd
before = pd.Timestamp.now()

results = got3.search_keyword_corpus('bovine', coca_corpus, 
                                            case_sensitive=False,
                                            show_context=True, 
                                            context_words=15,
                                            output='print',
                                            parallel=True)
after = pd.Timestamp.now()
print('time elapsed:', after - before)
```
```plaintext
🔍 COCA Corpus Search: 'bovine'
============================================================
🚀 Using parallel processing with 9 processes...

🎯 SUMMARY:
Total hits across all genre_years: 1196
Genre_years with matches: 206
time elapsed: 0 days 00:00:21.415171
```

tbd for more here ... 

______________________

______________________
### Legal Bert Text Masking

`getout_of_text3` provides a convenient interface to use these models for masked word prediction and other embedding tasks, namely using `got3.embedding.legal_bert.pipe()` function, `nlpaueb/legal-bert-base-uncased`, which is specifically trained on legal documents and is the most popular taged 'legal' on Hugging Face (https://huggingface.co/nlpaueb/legal-bert-base-uncased).

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
![https://raw.githubusercontent.com/atnjqt/getout_of_text_3/refs/heads/module-dev/img/legal_bert_bovine.png](https://raw.githubusercontent.com/atnjqt/getout_of_text_3/refs/heads/module-dev/img/legal_bert_bovine.png)

______________________
### EmbeddingGemma Document Similarity & Context Ranking

- The EmbeddingGemma model is designed for efficient text embeddings and can be used for various semantic tasks. The `got3.embedding.gemma.task()` function, leveraging `google/embeddinggemma-300m`, promises to be more efficient and effective across general text analysis (https://huggingface.co/google/embeddinggemma-300m) and is environmentally friendly in running AI on the device. The `got3` integrates large collections of keywords in context, documents, collocates, etc., allowing you to leverage this model for context ranking based on ambiguous terms in statutory languages. 

- The example below demonstrates how to use pre-computed search results from the COCA corpus to find the most relevant contexts for a given statutory phrasing.

- Other noteable examples include the latest
```python
### Trying it on got3
import getout_of_text_3 as got3

# First, perform a keyword search to get context data
# Use the new got3.embedding.gemma function with search results
result = got3.embedding.gemma.task(
    statutory_language="The agency may modify the requirements as necessary to ensure compliance.",
    ambiguous_term="modify",
    year_enacted=2001,
    search_results=keyword_list, # Pass the JSON results from search_keyword_corpus
    model="google/embeddinggemma-300m"
)
print('')
print("🎯 Top 3 most relevant contexts:")
for i, item in enumerate(result['all_ranked'][:3]):
    print(f"{i+1}. Genre: {item['genre']}, Score: {item['score']:.4f}")
    print(f"   Context: {item['context'][:100]}...")
    print()
```
```plaintext
📚 Using pre-computed search results for 'modify'
📚 Found 70 context examples across 7 genres
🤖 Loading model: google/embeddinggemma-300m

🎯 RESULTS:
Most relevant context from blog (score: 0.3598)
Context: is to enforce law created by Congress , not to **modify** it . Yes , he could have vetoed the reauthorization

🎯 Top 3 most relevant contexts:
1. Genre: blog, Score: 0.3598
   Context: is to enforce law created by Congress , not to **modify** it . Yes , he could have vetoed the reauth...

2. Genre: web, Score: 0.3385
   Context: standards : <p> Use existing Multi-Modal Level-of-Service indicators , and **modify** them to reflec...

3. Genre: news, Score: 0.3202
   Context: loan is going to foreclosure , it make sense to **modify** if you can get to the point where the bor...
```


____________________-

## Corpus of Contemporary American English (COCA)

You can gain access to the English Corpora website and register for academic / personal use of the COCA corpus at [https://www.english-corpora.org/coca/](https://www.english-corpora.org/coca/). If your university or organization has a subscription, you may be able to access the platform for free or at a reduced cost. Namely, your university or organization (probably the central library) may already have a subscription to the English Corpora dataset downloads and you can check with them to see if you can get access to the full dataset. You must agree to the terms of service and licensing agreement before downloading and using the COCA corpus files, which is namely to **not redistribute the corpus files** and to **not use the corpus for commercial purposes**.

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
    context_words=5,
    output="print" # json or print
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
print(freq_analysis['by_genre'][:2])  # preview

# Relative frequency per 10k tokens
freq_rel = got3.keyword_frequency_analysis(
    keyword="legal",
    db_dict=corpus_data,
    relative=True
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


## For Legal Researchers

This toolkit is specifically designed to support:

- **Constitutional Law Research** - Analyze constitutional language patterns across genres
- **Judicial Opinion Analysis** - Study linguistic patterns in legal decisions  
- **Legal Corpus Linguistics** - Examine legal language evolution over time
- **Comparative Legal Analysis** - Compare legal language usage across different contexts
- **Open Science Initiatives** - Enable reproducible legal research methodologies
- **Digital Humanities** - Support computational approaches to legal scholarship

## Documentation

- [API Reference](https://github.com/atnjqt/getout_of_text3) - Full function documentation

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

## Acknowledgements
We would like to thank the open-source community, legal scholars, and data scientists who have contributed to the development of this toolkit. Moreover, the UPenn Library Data Science team for their continued support.

**Advancing legal scholarship through open computational tools! ⚖️**

> **Disclaimer:** This project is still in development and may not yet be suitable for production use. The development of this project is heavily reliant on Artificial Intelligence coding tools for staging and deploying this PyPi module. Please use with caution as it is only intended for experimental use cases and explicitly provides no warranty of fitness for any particular task. In no way does this tool provide legal advice, nor do the authors of this module endorse any generative outputs you may observe or experience in using the toolset.
