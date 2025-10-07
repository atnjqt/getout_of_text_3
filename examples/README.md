# Examples

This directory contains example iPython Notebooks for demonstrative workflows using `getout_of_text3` across various corpora

## Overview

The examples directory includes several specialized directories for working with different text corpora:

- **[ai](./ai/)**: Examples of AI text analysis workflows
    - **[ai/reports](./ai/reports/)**: Generated analysis reports and visualizations from langchain agentic tools
- **[embedding](./embedding/)**: Examples of text embedding techniques
- **[topic_modeling](./topic_modeling/)**: Examples of topic modeling techniques

- **[coca](./coca/)**: Corpus of Contemporary American English examples
- **[scotus](./scotus/)**: Supreme Court of the United States text analysis

Additionally, the directory contains / assumes large data corpus json files:
- `coca_dict.json`: COCA dictionary data (5.75GB)
- `loc_gov.json`: Library of Congress data (394MB)

Each subdirectory contains iPython notebooks demonstrating specific analysis techniques and workflows for that corpus.

## Development

The following is the GitHub Copilot prompt used when beginning development on new examples or modules

```text
Hi AI, thanks for your help always I am appreciative. 
You are an expert in Python programming and NLP.
Your knowledge and skills in these areas are impressive.
--- 09/25/2025
I am working on a project that involves text analysis and I need your assistance.
Specifically, I am working on an open-source python module getout_of_text_3 which aims to provide transparency and interpretability in resolving the perrenial problem of ambiguious words in statuory interpretation, in the textualist debate of the US Supreme Court. I've already got a great start, with getting access to the full COCA dataset. I would like your help with implementing some specific features and functionalities in this module, as I originally did them for the sample dataset which it's filestructure was different dict. Let me know if you agree to help and I will provide you with details of what I need.
--- 10/02/2025
Currently I am looking to finalize the third portion of the toolset, namely AI tools, for use langchain and aws bedrock models to have an agent run some tooling.
In this case, I have a DIY corpus of US SCOTUS PDF extracts and I'm pulling keywords in context across the dataset, and then feeding to an AI model with a tool that is instructed to summarize the cases WITHOUT ever speculating or using info outside of what is provided in the prompt.
--- 10/07/2025
We have the working langchain tool in `./ai/langchain.ipynb` but I need to make it more robust and reliable by including it in the getout_of_text3 module as a class, and then making sure it works with the SCOTUS dataset. Namely this will be `got3.ai.agents.ScotusAnalysisTool` and `got3.ai.agents.ScotusFilteredAnalysisTool`. I'll give you all the files as context.
```