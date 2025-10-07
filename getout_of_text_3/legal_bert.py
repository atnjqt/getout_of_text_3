"""
Legal-BERT module for getout_of_text_3
Provides masked language modeling capabilities using Legal-BERT pipeline
"""

import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from transformers import pipeline
from typing import List, Dict, Optional, Any


# Global pipeline instance (lazy loaded)
_pipe = None


def _get_pipeline(model_name: str = "nlpaueb/legal-bert-base-uncased"):
    """Get or create the Legal-BERT pipeline"""
    global _pipe
    # If _pipe is None or model_name is different, create a new pipeline
    if _pipe is None or getattr(_pipe, 'model_name', None) != model_name:
        _pipe = pipeline("fill-mask", model=model_name)
        # Attach model_name attribute for future checks
        _pipe.model_name = model_name
    return _pipe


def pipe(statement: str, masked_token: Optional[str] = None, 
         token_mask: str = '[MASK]', top_k: int = 5, 
         visualize: bool = True, json_output: bool = False, model_name: str = "nlpaueb/legal-bert-base-uncased") -> List[Dict[str, Any]]:
    """
    Legal-BERT pipeline for masked language modeling
    
    Args:
        statement (str): The legal text with masked token(s)
        masked_token (str, optional): The actual token for display purposes
        token_mask (str): The mask token to replace (default: '[MASK]')
        top_k (int): Number of top predictions to return
        visualize (bool): Whether to show visualization (ignored if json_output=True)
        json_output (bool): If True, returns only JSON-serializable data without text/visualization
        
    Returns:
        List[Dict]: List of predictions with token_str, score, and other metadata
        
    Raises:
        ValueError: If token_mask is not found in statement
        
    Example:
        >>> import getout_of_text_3 as got3
        >>> statement = "The court ruled that the contract was [MASK]."
        >>> results = got3.embedding.legal_bert.pipe(statement, masked_token='valid')
        >>> # For JSON output only:
        >>> json_results = got3.embedding.legal_bert.pipe(statement, json_output=True)
    """
    if token_mask not in statement:
        raise ValueError(f"The token_mask '{token_mask}' is not in the statement.")
    
    # Get the pipeline and make predictions
    legal_bert_pipeline = _get_pipeline(model_name=model_name)
    results = legal_bert_pipeline(statement, top_k=top_k)
    
    # If JSON output requested, return raw results without any display
    if json_output:
        return results
    
    # Always print the text results (unless JSON mode)
    _print_predictions(results)
    
    # Show visualization only if requested
    if visualize:
        _visualize_predictions(results, statement, masked_token, token_mask, show_text=False)
    
    return results


def _print_predictions(results: List[Dict]) -> None:
    """Print prediction results in a formatted way"""
    # Extract token strings and scores, sort by score descending
    tokens = [result['token_str'] for result in results]
    scores = [result['score'] for result in results]
    
    # Create DataFrame and sort by score descending
    df = pd.DataFrame({'token': tokens, 'score': scores})
    df = df.sort_values('score', ascending=False)
    
    # Print results
    print("Top predictions for masked token (highest to lowest):")
    for i, (_, row) in enumerate(df.iterrows(), 1):
        print(f"{i}. '{row['token']}' - Score: {row['score']:.4f}")


def _visualize_predictions(results: List[Dict], statement: str, 
                         masked_token: Optional[str] = None, 
                         token_mask: str = '[MASK]', 
                         figsize: tuple = (8, 4),
                         show_text: bool = True) -> None:
    """Create a visualization of masked token predictions"""
    # Extract token strings and scores
    tokens = [result['token_str'] for result in results]
    scores = [result['score'] for result in results]
    
    # Create DataFrame and sort by score descending
    df = pd.DataFrame({'token': tokens, 'score': scores})
    df = df.sort_values('score', ascending=False)
    
    # Set modern style
    sns.set_palette("viridis")
    
    # Create a horizontal bar plot
    plt.figure(figsize=figsize)
    bars = plt.barh(df['token'], df['score'], 
                   color=sns.color_palette("viridis", len(df)), 
                   edgecolor='white', linewidth=0.8)
    
    plt.xlabel('Prediction Score', fontsize=11)
    plt.ylabel('Predicted Tokens', fontsize=11)
    plt.title('Legal-BERT Masked Token Predictions', fontsize=12, fontweight='bold', pad=15)
    
    # Add subtitle with the statement
    if masked_token:
        display_statement = statement.replace(token_mask, f' [{masked_token}] ')
    else:
        display_statement = statement
        
    plt.suptitle(f"Statement: {display_statement}", 
                fontsize=10, fontweight='bold', y=-0.05, color='blue')
    
    # Add score labels on the bars
    for i, (token, score) in enumerate(zip(df['token'], df['score'])):
        plt.text(score + 0.005, i, f'{score:.3f}', 
                va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Print results only if show_text is True
    if show_text:
        _print_predictions(results)


def get_best_prediction(statement: str, token_mask: str = '[MASK]') -> Dict[str, Any]:
    """Get the top prediction for a masked token
    
    Args:
        statement (str): The legal text with masked token(s)
        token_mask (str): The mask token to replace (default: '[MASK]')
        
    Returns:
        Dict: The top prediction with token_str, score, and other metadata
    """
    results = pipe(statement, token_mask=token_mask, top_k=1, visualize=False)
    return results[0] if results else None


# Legacy compatibility - redirect to pipe function
def legal_bert(*args, **kwargs):
    """Legacy function - redirects to pipe()"""
    return pipe(*args, **kwargs)


def to_json(results: List[Dict[str, Any]], indent: int = 2) -> str:
    """
    Convert prediction results to JSON string
    
    Args:
        results (List[Dict]): The prediction results from pipe()
        indent (int): JSON indentation level
        
    Returns:
        str: JSON-formatted string of the results
        
    Example:
        >>> results = got3.embedding.legal_bert.pipe(statement, json_output=True)
        >>> json_str = got3.embedding.legal_bert.to_json(results)
        >>> print(json_str)
    """
    # Clean up results for JSON serialization
    clean_results = []
    for result in results:
        clean_result = {
            'token': result.get('token_str', ''),
            'score': float(result.get('score', 0.0)),
            'token_id': int(result.get('token', 0)),
            'sequence': result.get('sequence', '')
        }
        clean_results.append(clean_result)
    
    return json.dumps(clean_results, indent=indent, ensure_ascii=False)
