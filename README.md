# getout-of-text-3: A Python Toolkit for Legal Text Analysis and Open Science

- Author: Etienne P Jacquot (`@atnjqt`)

## Introduction

The `getout_of_text3` module is a comprehensive Python library promoting open and reproducible computational forensic linguistics toolsets for data scientists and legal scholars performing textual analysis with popular corpora such as **COCA** ([Corpus of Contemporary American English](https://www.english-corpora.org/coca/)), **SCOTUS** [Library of Congress US Report Collection](https://www.loc.gov/collections/united-states-reports/), and other legal / natural language text corpora by providing simpler toolsets to promote the discovery of the *'ordinary meaning'* of words using NLP, Embedding Models, and AI Agentic LLMs.

### Installation

You can install `getout_of_text3` using pip. I recommend setting up a virtual environment using [venv](https://docs.python.org/3/library/venv.html) or [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) to manage dependencies.

```bash
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate
pip install getout-of-text-3 -U
```


## Table of Contents

- [Overview](#overview)
    - [Key Features for Legal & Linguistic Scholars](#key-features-for-legal--linguistic-scholars)
- [Getting Started](#getting-started)
    - [Installation](#installation)
    - [Corpus of Contemporary American English (COCA)](#corpus-of-contemporary-american-english-coca)
        - [Genres & Years](#genres--years)
        - [Read the Dataset](#read-the-dataset)
        - [Search for Keyword in Context](#search-for-keyword-in-context)
        - [Search for Keyword Distribution across Genres](#search-for-keyword-distribution-across-genres)
- [NLP](#nlp)
    - [Text Preprocessing](#text-preprocessing)
- [AI Agents](#ai-agents)
    - [Langchain & AWS Bedrock](#langchain--aws-bedrock)
- [Embedding Models](#embedding-models)
    - [Legal Bert Text Masking](#legal-bert-text-masking)
    - [EmbeddingGemma Document Similarity & Context Ranking](#embeddinggemma-document-similarity--context-ranking)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)
- [Support](#support)
- [Acknowledgements](#acknowledgements)

## Overview

The `got3` module aims to provide simpler toolsets to promote the discovery of the *'ordinary meaning'* of words in and out of legal contexts using computational techniques, with a focus on delivering an open-source tool built around three main areas of functionality:

### Key Features for Legal & Linguistic Scholars

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

The below examples demonstrate how to use the `getout_of_text3` module for various tasks using corpus linguistics tools, embedding models, and AI language models.

______________________

### Corpus of Contemporary American English (COCA)

If you have access and paid for the COCA corpus (https://www.corpusdata.org/purchase.asp with academic & commercial licenses), you can use the `got3` module to read and analyze the corpus files. Please ensure you comply with the licensing terms of COCA when using the corpus data.

> 📝 Note: The COCA corpus is a large and diverse corpus of American English, and it DOES contain sensitive or proprietary information. Please use the corpus responsibly and in accordance with the licensing terms. English Corpora scrubs 95%/5% with 10 `@` signs, so you may notice that in search results as an effort to promote fair use doctrine in copyright law. The database maintainers also add a watermark throughtout the text content that deviates from the real content, and periodically scan the public web for distribution of this content. You must agree to the terms of service and licensing agreement before downloading and using the COCA corpus files, which is namely to **not redistribute the corpus files** and to **not use the corpus for commercial purposes**.


#### Genres & Years

Years are from 1990-2019 for the following distributions of COCA:

1. Academic (`acad`) - Legal academic texts
2. Blog (`blog`) - Legal blogs and commentary  
3. Fiction (`fic`) - Legal fiction and narratives
4. Magazine (`mag`) - Legal magazine articles
5. News (`news`) - Legal news coverage
6. Spoken (`spok`) - Legal oral arguments and speeches
7. TV/Movie (`tvm`) - Legal drama and media
8. Web (`web`) - Legal web content

#### Read the Dataset

- the `./coca-text/` directory should contain the COCA text files you downloaded from the English Corpora website, such as `text_acad.txt`, `text_blog.txt`, etc. It's organized by genre and year, except for Web & Blog that are by index.

```python
### Trying it on got3
import getout_of_text_3 as got3

coca_corpus = got3.read_corpus('./coca-text/')
```

#### Search for Keyword in Context

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

#### Search for Keyword Distribution across Genres

- get a distribution of a keyword across genres, for example still using `bovine`:

```python
before = pd.Timestamp.now()
bovine_freq = got3.keyword_frequency_analysis('bovine', 
                                              coca_corpus, 
                                              case_sensitive=False,
                                              relative=True, # optionally to show column, per 10k words
                                              parallel=True # use parallel processing
                                              )
after = pd.Timestamp.now()
print('time elapsed:', after - before)
```
```plaintext
📊 Frequency Analysis for 'bovine' (case_sensitive=False, loose substring match)
============================================================
  acad    :    501 hits | 140449282 tokens | 0.04 /10k
  web     :    208 hits | 149036464 tokens | 0.01 /10k
  mag     :    162 hits | 146417442 tokens | 0.01 /10k
  fic     :    109 hits | 142585624 tokens | 0.01 /10k
  blog    :     92 hits | 143156927 tokens | 0.01 /10k
  tvm     :     71 hits | 162287598 tokens | 0.00 /10k
  news    :     68 hits | 143377305 tokens | 0.00 /10k
  spok    :     41 hits | 151501397 tokens | 0.00 /10k
------------------------------------------------------------
TOTAL: 1252 hits across 8 genres (~1178812039 tokens)
time elapsed: 0 days 00:00:35.216885
```


______________________

## NLP

### Text Preprocessing

KWIC, Collocates, Frequency Analysis, etc.

## AI Agents

### Langchain & AWS Bedrock

`getout_of_text3` can provide filtered results to pass to AI agents for further analysis, summarization, or technical steps in a toolchain. TBD as I've not yet implemented this in the toolset but examples are provided in `examples/ai/demo.ipynb` reference provided.

> 🚨 This requires an AWS `named_profile` and will incur marginal costs! TBD on future versions supporting AI agents beyond AWS Bedrock.


```python
import pandas as pd
import getout_of_text_3 as got3
from getout_of_text_3 import ScotusAnalysisTool, ScotusFilteredAnalysisTool
from langchain.chat_models import init_chat_model

model = init_chat_model(
    "openai.gpt-oss-120b-1:0",
    model_provider="bedrock_converse",
    credentials_profile_name="atn-developer",
    max_tokens=128000
)
# Assume you built db_dict_formatted (volume -> DataFrame with columns ['case_id','text'])
search_tool = ScotusAnalysisTool(model=model, db_dict_formatted=db_dict_formatted)
filtered_tool = ScotusFilteredAnalysisTool(model=model)

# Quick term filter & summarization
text_result = search_tool._run(keyword="bank", 
                               analysis_focus="general")
```

![alt text](img/ai_agent_example_bank.png)

## Embedding Models

### Legal Bert Text Masking

`getout_of_text3` provides a convenient interface to use these models for masked word prediction and other embedding tasks, namely using `got3.embedding.legal_bert.pipe()` function, `nlpaueb/legal-bert-base-uncased`, which is specifically trained on legal documents and is the most popular taged 'legal' on Hugging Face (https://huggingface.co/nlpaueb/legal-bert-base-uncased).

```python
### Trying it on got3
import getout_of_text_3 as got3

statement = "Establishing a system for the identification and registration of [MASK] animals and regarding the labelling of beef and beef products."
masked_token="bovine"
token_mask="[MASK]"

got3.embedding.legal_bert.pipe(statement=statement, # the input text with a [MASK] token
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



## Documentation

- [API Reference](https://github.com/atnjqt/getout_of_text3) - Full function documentation

## Contributing

We welcome contributions from legal scholars and developers! Please see our contributing guidelines and feel free to submit issues or pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.

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
