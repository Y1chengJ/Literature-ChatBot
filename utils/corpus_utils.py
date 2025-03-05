import pickle
import os
import logging
from typing import Dict, Any
from datasets import load_from_disk

logger = logging.getLogger(__name__)

def load_or_process_corpus(corpus_path: str, pickle_path: str = "corpus.pkl") -> Dict[str, Dict[str, Any]]:
    """
    Load processed corpus from pickle file, or process dataset and save as pickle if it doesn't exist
    
    Parameters:
        corpus_path: str - Path to the dataset
        pickle_path: str - Path to save/load pickle file
        
    Returns:
        Processed corpus dictionary
    """
    if os.path.exists(pickle_path):
        logger.info(f"Loading corpus from pickle: {pickle_path}")
        with open(pickle_path, 'rb') as f:
            return pickle.load(f)
    
    # If pickle doesn't exist, process from dataset
    logger.info(f"Processing corpus from dataset: {corpus_path}")
    dataset = load_from_disk(corpus_path)
    
    # Convert to required format
    corpus = {}
    for item in dataset:
        doc_id = str(item['id'])
        corpus[doc_id] = {
            'title': item['title'],
            'abstract': item.get('abstract', ''),
            'authors': item.get('authors', ''),
            'published_date': item.get('published_date', ''),
            'arxiv_id': item.get('arxiv_id', '')
        }
    
    # Save processed corpus
    logger.info(f"Saving processed corpus to pickle: {pickle_path}")
    with open(pickle_path, 'wb') as f:
        pickle.dump(corpus, f)
    
    return corpus
