"""
EmbeddingGemma module for getout_of_text_3
Provides embedding-based question answering capabilities using EmbeddingGemma model
"""

import json
from typing import List, Dict, Optional, Any, Tuple

def task(statutory_language: str, ambiguous_term: str, year_enacted: Optional[int] = None, 
          model: str = "google/embeddinggemma-300m", search_results: Optional[Dict] = None,
          corpus_data: Optional[Dict] = None, context_words: int = 5, **kwargs) -> Dict[str, Any]:
    """
    EmbeddingGemma-based question answering for statutory language interpretation
    
    This function combines corpus search with embedding-based semantic similarity to help
    interpret ambiguous terms in statutory language by finding the most relevant contexts
    from the corpus data.
    
    Args:
        statutory_language (str): The statutory text containing the ambiguous term
        ambiguous_term (str): The term that needs interpretation
        year_enacted (int, optional): Year the statute was enacted for historical context
        model (str): EmbeddingGemma model to use (default: google/embeddinggemma-300m)
        search_results (dict, optional): Pre-computed JSON results from got3.search_keyword_corpus(output='json')
        corpus_data (dict, optional): Raw corpus data. If None and search_results None, will raise error
        context_words (int): Number of context words around the term (default: 5)
        **kwargs: Additional arguments for model configuration
        
    Returns:
        Dict containing:
        - query: The question asked
        - statutory_language: Original statutory text
        - ambiguous_term: The term being interpreted
        - year_enacted: Year of enactment
        - most_relevant: Most relevant document/context
        - all_ranked: All documents ranked by relevance
        - model_used: Model identifier
        - corpus_stats: Statistics about corpus search
        
    Example:
        >>> import getout_of_text_3 as got3
        >>> corpus_data = got3.read_corpora("corpus-files/", "legal_corpus")
        >>> results = got3.search_keyword_corpus("bank", corpus_data, output="json")
        >>> result = got3.embedding.gemma.task(
        ...     statutory_language="The bank shall maintain sufficient reserves.",
        ...     ambiguous_term="bank",
        ...     year_enacted=2024,
        ...     search_results=results
        ... )
        >>> print(result['most_relevant'])
    """
    try:
        from sentence_transformers import SentenceTransformer
        import torch
    except ImportError:
        raise ImportError(
            "EmbeddingGemma dependencies not installed. Run: "
            "pip install sentence-transformers torch"
        )
    
    # Import corpus functionality
    from .corpus import LegalCorpus
    
    # Handle input: either search_results (JSON) or corpus_data (raw)
    if search_results is not None:
        # Use pre-computed search results
        print(f"📚 Using pre-computed search results for '{ambiguous_term}'")
    elif corpus_data is not None:
        # Search for the ambiguous term in the corpus
        corpus = LegalCorpus()
        print(f"🔍 Searching for '{ambiguous_term}' in corpus...")
        search_results = corpus.search_keyword_corpus(
            keyword=ambiguous_term,
            db_dict=corpus_data,
            case_sensitive=False,
            show_context=True,
            context_words=context_words,
            output='json'
        )
    else:
        raise ValueError(
            "Either search_results or corpus_data is required. "
            "Use got3.search_keyword_corpus(output='json') to get search_results, "
            "or got3.read_corpora() to get corpus_data."
        )
    
    # Convert search results to documents for embedding
    documents = []
    document_metadata = []
    
    for genre, texts in search_results.items():
        for text_id, context in texts.items():
            # Format as title | text for better embedding performance
            doc = f"title: {genre.upper()} Text {text_id} | text: {context}"
            documents.append(doc)
            document_metadata.append({
                'genre': genre,
                'text_id': text_id,
                'context': context
            })
    
    if not documents:
        return {
            'error': f"No occurrences of '{ambiguous_term}' found in corpus",
            'query': None,
            'statutory_language': statutory_language,
            'ambiguous_term': ambiguous_term,
            'year_enacted': year_enacted,
            'corpus_stats': {'total_documents': 0, 'genres': []}
        }
    
    print(f"📚 Found {len(documents)} context examples across {len(search_results)} genres")
    
    # Build the question
    if year_enacted:
        query = (
            f'What is the ordinary meaning of the ambiguous term "{ambiguous_term}" '
            f'in the context of the following statutory language, "{statutory_language}", '
            f'enacted in the year {year_enacted}?'
        )
    else:
        query = (
            f'What is the ordinary meaning of the ambiguous term "{ambiguous_term}" '
            f'in the context of the following statutory language, "{statutory_language}"?'
        )
    
    print(f"🤖 Loading model: {model}")
    
    # Load the model
    embedding_model = SentenceTransformer(model)
    
    # Encode query and documents
    query_prompt = f"task: question answering | query: {query}"
    query_embedding = embedding_model.encode_query(query_prompt)
    document_embeddings = embedding_model.encode_document(documents)
    
    # Calculate similarities
    similarities = embedding_model.similarity(query_embedding, document_embeddings)
    
    # Get rankings
    ranked_indices = torch.argsort(similarities, descending=True)[0]
    
    # Prepare results
    most_relevant_idx = ranked_indices[0].item()
    most_relevant_doc = documents[most_relevant_idx]
    most_relevant_meta = document_metadata[most_relevant_idx]
    
    all_ranked = []
    for i, idx in enumerate(ranked_indices):
        idx = idx.item()
        all_ranked.append({
            'rank': i + 1,
            'score': float(similarities[0][idx]),
            'document': documents[idx],
            'genre': document_metadata[idx]['genre'],
            'text_id': document_metadata[idx]['text_id'],
            'context': document_metadata[idx]['context']
        })
    
    # Calculate corpus statistics
    genre_counts = {}
    for meta in document_metadata:
        genre = meta['genre']
        genre_counts[genre] = genre_counts.get(genre, 0) + 1
    
    print(f"\n🎯 RESULTS:\n")
    print(f"Most relevant context from {most_relevant_meta['genre']} (score: {float(similarities[0][most_relevant_idx]):.4f})")
    print(f"Context: {most_relevant_meta['context']}")
    
    return {
        'query': query,
        'statutory_language': statutory_language,
        'ambiguous_term': ambiguous_term,
        'year_enacted': year_enacted,
        'most_relevant': {
            'score': float(similarities[0][most_relevant_idx]),
            'document': most_relevant_doc,
            'genre': most_relevant_meta['genre'],
            'text_id': most_relevant_meta['text_id'],
            'context': most_relevant_meta['context']
        },
        'all_ranked': all_ranked,
        'model_used': model,
        'corpus_stats': {
            'total_documents': len(documents),
            'genres': list(genre_counts.keys()),
            'genre_distribution': genre_counts
        }
    }


def to_json(results: Dict[str, Any], indent: int = 2) -> str:
    """
    Convert task results to JSON string
    
    Args:
        results (Dict): The results from task()
        indent (int): JSON indentation level
        
    Returns:
        str: JSON-formatted string of the results
        
    Example:
        >>> result = got3.embedding.gemma.task(statutory_language, ambiguous_term, corpus_data=corpus)
        >>> json_str = got3.embedding.gemma.to_json(result)
        >>> print(json_str)
    """
    return json.dumps(results, indent=indent, ensure_ascii=False, default=str)


def get_top_contexts(results: Dict[str, Any], n: int = 5) -> List[Dict[str, Any]]:
    """
    Get the top N most relevant contexts from task results
    
    Args:
        results (Dict): The results from task()
        n (int): Number of top contexts to return
        
    Returns:
        List[Dict]: Top N contexts with metadata
    """
    if 'error' in results:
        return []
    
    return results.get('all_ranked', [])[:n]


def get_genre_analysis(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze results by genre to see which types of texts are most relevant
    
    Args:
        results (Dict): The results from task()
        
    Returns:
        Dict: Genre-based analysis of relevance
    """
    if 'error' in results:
        return {'error': results['error']}
    
    genre_scores = {}
    genre_counts = {}
    
    for item in results.get('all_ranked', []):
        genre = item['genre']
        score = item['score']
        
        if genre not in genre_scores:
            genre_scores[genre] = []
            genre_counts[genre] = 0
        
        genre_scores[genre].append(score)
        genre_counts[genre] += 1
    
    # Calculate average scores per genre
    genre_averages = {}
    for genre, scores in genre_scores.items():
        genre_averages[genre] = {
            'average_score': sum(scores) / len(scores),
            'max_score': max(scores),
            'count': len(scores),
            'scores': scores
        }
    
    # Sort by average score
    sorted_genres = sorted(
        genre_averages.items(), 
        key=lambda x: x[1]['average_score'], 
        reverse=True
    )
    
    return {
        'genre_analysis': dict(sorted_genres),
        'most_relevant_genre': sorted_genres[0][0] if sorted_genres else None,
        'corpus_stats': results.get('corpus_stats', {})
    }
