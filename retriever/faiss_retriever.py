import os
import logging
import pickle
import torch
import numpy as np
from typing import Dict, List, Union, Optional, Type
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Import FAISS implementations
import faiss
from retriever.faiss_index import (
    FaissIndex, 
    FaissHNSWIndex, 
    FaissTrainIndex, 
    FaissBinaryIndex
)
from retriever.faiss_search import (
    DenseRetrievalFaissSearch,
    FlatIPFaissSearch,
    HNSWFaissSearch,
    HNSWSQFaissSearch,
    PQFaissSearch,
    SQFaissSearch,
    BinaryFaissSearch
)

logger = logging.getLogger(__name__)

class FaissRetriever:
    """
    A retriever implementation that uses FAISS for efficient similarity search.
    """
    
    def __init__(
        self, 
        corpus: Dict[str, Dict[str, str]] = None,
        model_name: str = "intfloat/multilingual-e5-large-instruct",
        index_type: str = "flat",
        batch_size: int = 32,
        embeddings_dir: str = "faiss_embeddings",
        use_gpu: bool = torch.cuda.is_available()
    ):
        """
        Initialize the FAISS retriever
        
        Args:
            corpus: Document corpus dictionary {doc_id: {title, abstract, ...}}
            model_name: Name of the embedding model to use
            index_type: Type of FAISS index ("flat", "hnsw", "pq", "sq", "hnswsq", "binary")
            batch_size: Batch size for encoding
            embeddings_dir: Directory to save/load embeddings and indices
            use_gpu: Whether to use GPU for search (if available)
        """
        self.corpus = corpus
        self.model_name = model_name
        self.index_type = index_type.lower()
        self.batch_size = batch_size
        self.embeddings_dir = embeddings_dir
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        # Load model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_model = SentenceTransformer(model_name, device=self.device)
        
        # Initialize FAISS search implementation based on index type
        self._init_faiss_search()
        
        # If corpus is provided, initialize embeddings and index
        if corpus is not None:
            self.initialize(embeddings_dir)
    
    def _init_faiss_search(self):
        """Initialize the appropriate FAISS search implementation"""
        # Map of index types to their search implementation classes
        index_map = {
            "flat": FlatIPFaissSearch,
            "hnsw": HNSWFaissSearch,
            "pq": PQFaissSearch,
            "sq": SQFaissSearch,
            "hnswsq": HNSWSQFaissSearch,
            "binary": BinaryFaissSearch,
        }
        
        if self.index_type not in index_map:
            valid_types = ", ".join(index_map.keys())
            raise ValueError(f"Invalid index type: {self.index_type}. Must be one of: {valid_types}")
            
        # Create search implementation
        self.search_impl = index_map[self.index_type](
            model=self.embedding_model,
            batch_size=self.batch_size,
            use_gpu=self.use_gpu
        )
    
    def initialize(self, save_dir: str = None):
        """Initialize embeddings and index"""
        if save_dir is None:
            save_dir = self.embeddings_dir
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Try to load existing index
        index_file = os.path.join(save_dir, f"faiss_{self.index_type}.faiss")
        mappings_file = os.path.join(save_dir, f"faiss_{self.index_type}.tsv")
        
        if os.path.exists(index_file) and os.path.exists(mappings_file):
            logger.info(f"Loading existing FAISS index from {index_file}")
            self.search_impl.load(save_dir, prefix="faiss", ext=self.index_type)
            logger.info("FAISS index loaded successfully")
        else:
            # Index needs to be built
            if self.corpus is None:
                raise ValueError("No corpus provided and no existing index found")
            
            logger.info(f"Building new {self.index_type} FAISS index")
            self.build_index(save_dir=save_dir)
    
    def build_index(self, save_dir: str = None):
        """Build the FAISS index from the corpus"""
        if save_dir is None:
            save_dir = self.embeddings_dir
            
        os.makedirs(save_dir, exist_ok=True)
        
        # Index the corpus
        logger.info("Indexing corpus with FAISS...")
        self.search_impl.index(self.corpus, score_function="cos_sim")
        
        # Save the index
        logger.info(f"Saving index to {save_dir}")
        self.search_impl.save(save_dir, prefix="faiss", ext=self.index_type)
        
        logger.info("FAISS index built and saved successfully")
    
    def search(self, queries: Dict[str, str], top_k: int = 5) -> Dict[str, Dict[str, float]]:
        """
        Search for documents similar to the queries
        
        Args:
            queries: Dictionary mapping query_id to query_text
            top_k: Number of top results to return
            
        Returns:
            Dictionary mapping query_id to {doc_id: similarity_score} pairs
        """
        if self.search_impl is None:
            raise ValueError("Search implementation not initialized")
            
        # Perform search
        return self.search_impl.search(
            corpus=self.corpus,
            queries=queries,
            top_k=top_k,
            score_function="cos_sim"
        )
    
    def get_corpus_size(self):
        """Get the size of the corpus"""
        return len(self.corpus) if self.corpus else 0
        
    def get_index_type(self):
        """Get the type of FAISS index being used"""
        return self.index_type
    
    def get_index_name(self):
        """Get the name of the FAISS index"""
        return self.search_impl.get_index_name() if self.search_impl else None


def main():
    """Test the FAISS retriever"""
    logging.basicConfig(level=logging.INFO)
    
    # Load corpus from processed dataset
    from utils.corpus_utils import load_or_process_corpus
    import time
    
    logger.info("Loading corpus...")
    corpus = load_or_process_corpus(corpus_path="arxiv_corpus", pickle_path="corpus.pkl")
    
    # Test queries
    queries = {
        "q1": "large language models",
        "q2": "neural networks",
        "q3": "machine learning"
    }
    
    # Test different FAISS index types
    index_types = ["flat", "hnsw"]
    
    for idx_type in index_types:
        logger.info(f"\n--- Testing {idx_type.upper()} index ---")
        
        # Create retriever
        retriever = FaissRetriever(
            corpus=corpus,
            index_type=idx_type,
            embeddings_dir=f"faiss_embeds_{idx_type}"
        )
        
        # Search
        start_time = time.time()
        results = retriever.search(queries, top_k=5)
        search_time = time.time() - start_time
        
        logger.info(f"Search with {idx_type} took {search_time:.4f} seconds")
        
        # Print results
        print(f"\n{idx_type.upper()} Search Results:")
        for query_id, docs in results.items():
            print(f"\nQuery: {queries[query_id]}")
            for i, (doc_id, score) in enumerate(list(docs.items())[:3]):
                doc = corpus[doc_id]
                print(f"Result {i+1}: {doc['title']} (Score: {score:.4f})")
            
        print("-" * 80)


if __name__ == "__main__":
    main()
