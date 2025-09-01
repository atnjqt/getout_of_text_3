#pip install getout-of-text-3==0.1.2

import getout_of_text_3 as got3


print(got3.__version__)

keyword="textual"

# 1. Read COCA corpus files
corpus_data = got3.read_corpora("./coca-samples-text", "coca")

# 2. Search for legal terms with context
results = got3.search_keyword_corpus(
    keyword=keyword,
    db_dict=corpus_data,
    case_sensitive=False,
    show_context=True,
    context_words=5
)

print(results)

# 3. Find collocates (words that appear near your target term)
collocates = got3.find_collocates(
    keyword=keyword,
    db_dict=corpus_data,
    window_size=5,
    min_freq=2
)

print(collocates)

# 4. Analyze frequency across genres
freq_analysis = got3.keyword_frequency_analysis(
    keyword=keyword,
    db_dict=corpus_data
)