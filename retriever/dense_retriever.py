import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Union, Optional
import logging
from tqdm import tqdm
import os
import faiss
import pickle


logger = logging.getLogger(__name__)

class DenseRetriever:
    def __init__(self, model_name: str = "intfloat/multilingual-e5-large-instruct", batch_size: int = 32, 
                 corpus: Dict[str, Dict[str, str]] = None, embeddings_dir: str = "./data/embeddings", 
                 use_faiss: bool = False):
        """
        Initialize retriever
        
        Parameters:
            model_name: Name of the embedding model
            batch_size: Batch size for processing
            corpus: Document corpus
            embeddings_dir: Directory for saving embeddings
            use_faiss: Whether to use FAISS for vector search
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        self.corpus_embeddings = None
        self.corpus = corpus
        self.corpus_ids = None if corpus is None else list(corpus.keys())
        self.use_faiss = use_faiss
        self.faiss_index = None
        
        # If corpus is provided, automatically process embeddings
        self.embedding_save_dir = embeddings_dir
        if corpus is not None:
            self.initialize_embeddings()
    
    def initialize_embeddings(self):
        """Initialize embeddings: load if exists, otherwise encode and save"""
        logger.info(f"Try to load embeddings from {os.path.abspath(self.embedding_save_dir)}")

        if self.load_vectors(self.embedding_save_dir):
            logger.info("Embeddings loaded successfully.")
            # If using FAISS, initialize the index
            if self.use_faiss and self.corpus_embeddings is not None:
                self._init_faiss_index()
        else:
            # If loading fails and corpus exists, encode corpus
            if self.corpus is not None:
                logger.info("No existing embeddings found. Encoding corpus...")
                self.encode_corpus()
                logger.info("Corpus encoding completed.")
                # Initialize FAISS index after encoding
                if self.use_faiss:
                    self._init_faiss_index()
            else:
                logger.warning("No existing embeddings and no corpus provided.")

    def encode_corpus(self):
        """Encode corpus and save vectors"""
        if self.corpus is None:
            raise ValueError("No corpus available for encoding. Please provide a corpus.")
            
        logger.info("Encoding corpus...")
        
        # Prepare documents
        texts = []
        
        # Combine title and abstract
        for doc_id in self.corpus_ids:
            doc = self.corpus[doc_id]
            text = f"{doc['title']} {doc.get('abstract', '')}"
            texts.append(text)
        
        # Batch encoding
        logger.info("Computing embeddings...")
        all_embeddings = []
        for i in tqdm(range(0, len(texts), self.batch_size)):
            batch_texts = texts[i:i + self.batch_size]
            embeddings = self.model.encode(
                batch_texts,
                convert_to_tensor=True,
                show_progress_bar=False
            )
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings)
        
        self.corpus_embeddings = torch.cat(all_embeddings)
        
        # Save vectors
        os.makedirs(self.embedding_save_dir, exist_ok=True)
        vector_path = os.path.join(self.embedding_save_dir, "corpus_vectors.pt")
        torch.save(self.corpus_embeddings, vector_path)
        logger.info(f"Saved vectors to {vector_path}")
        
        # Initialize FAISS index if needed
        if self.use_faiss:
            self._init_faiss_index()
            self._save_faiss_index(self.embedding_save_dir)

    def load_vectors(self, save_dir: str = "./data/embeddings") -> bool:
        """Load saved vectors, return whether loading was successful"""
        vector_path = os.path.join(save_dir, "corpus_vectors.pt")
        if os.path.exists(vector_path):
            logger.info(f"Loading vectors from {vector_path}...")
            self.corpus_embeddings = torch.load(vector_path)
            
            # Try to load FAISS index if use_faiss is True
            if self.use_faiss:
                faiss_path = os.path.join(save_dir, "faiss_index.bin")
                if os.path.exists(faiss_path):
                    self._load_faiss_index(save_dir)
                else:
                    logger.info("FAISS index not found. Creating new index...")
                    self._init_faiss_index()
                    self._save_faiss_index(save_dir)
                
            return True
        else:
            logger.warning("No saved vectors found.")
            return False
    
    def _init_faiss_index(self):
        """Initialize FAISS index from the corpus embeddings"""
        if self.corpus_embeddings is None:
            raise ValueError("No corpus embeddings available for FAISS indexing")
        
        logger.info("Initializing FAISS index...")
        # Convert to numpy for FAISS
        embeddings_np = self.corpus_embeddings.cpu().numpy()
        vector_dim = embeddings_np.shape[1]
        
        # Use L2 normalization with cosine distance
        # For cosine similarity, we use Inner Product because vectors are normalized
        self.faiss_index = faiss.IndexFlatIP(vector_dim)
        # Add vectors to the index
        self.faiss_index.add(embeddings_np)
        logger.info(f"FAISS index created with {self.faiss_index.ntotal} vectors")
    
    def _save_faiss_index(self):
        """Save FAISS index to disk"""
        if self.faiss_index is None:
            logger.warning("No FAISS index to save")
            return
        
        os.makedirs(self.embedding_save_dir, exist_ok=True)
        faiss_path = os.path.join(self.embedding_save_dir, "faiss_index.bin")
        
        # Save FAISS index
        faiss.write_index(self.faiss_index, faiss_path)
        logger.info(f"Saved FAISS index to {faiss_path}")
        
        # Also save corpus_ids mapping to correctly interpret FAISS results
        ids_path = os.path.join(self.embedding_save_dir, "corpus_ids.pkl")
        with open(ids_path, 'wb') as f:
            pickle.dump(self.corpus_ids, f)
        logger.info(f"Saved corpus IDs to {ids_path}")
    
    def _load_faiss_index(self, save_dir: str) -> bool:
        """Load FAISS index from disk"""
        faiss_path = os.path.join(save_dir, "faiss_index.bin")
        ids_path = os.path.join(save_dir, "corpus_ids.pkl")
        
        if os.path.exists(faiss_path) and os.path.exists(ids_path):
            logger.info(f"Loading FAISS index from {faiss_path}...")
            self.faiss_index = faiss.read_index(faiss_path)
            
            # Load corpus_ids if needed
            if self.corpus_ids is None:
                with open(ids_path, 'rb') as f:
                    self.corpus_ids = pickle.load(f)
                    
            logger.info(f"FAISS index loaded with {self.faiss_index.ntotal} vectors")
            return True
        else:
            logger.warning("FAISS index or corpus IDs file not found.")
            return False
    
    def _search_with_torch(self, query_embeddings: torch.Tensor, top_k: int) -> tuple:
        """Search documents using PyTorch"""
        # Calculate similarity
        similarity = torch.mm(query_embeddings, self.corpus_embeddings.t())
        
        # Get top-k results
        top_k_values, top_k_indices = torch.topk(
            similarity, k=min(top_k, self.corpus_embeddings.size(0)), dim=1
        )
        
        return top_k_values, top_k_indices
    
    def _search_with_faiss(self, query_embeddings: np.ndarray, top_k: int) -> tuple:
        """Search documents using FAISS"""
        if self.faiss_index is None:
            raise ValueError("FAISS index not initialized. Call _init_faiss_index() first.")
        
        # FAISS search
        scores, indices = self.faiss_index.search(query_embeddings, k=top_k)
        
        # Convert to torch tensors for consistency
        scores_tensor = torch.from_numpy(scores)
        indices_tensor = torch.from_numpy(indices)
        
        return scores_tensor, indices_tensor

    def search(self, queries: Dict[str, str], top_k: int = 5, 
               use_faiss: Optional[bool] = None) -> Dict[str, Dict[str, float]]:
        """
        Search for similar documents
        
        Parameters:
            queries: Dictionary of query_id to query_text
            top_k: Number of top results to return
            use_faiss: Whether to use FAISS for search (overrides instance setting)
            
        Returns:
            Dictionary of query_id to {doc_id: similarity_score} mappings
        """
        if self.corpus_embeddings is None:
            raise ValueError("No corpus embeddings available. Please initialize embeddings first.")
        
        if self.corpus is None:
            raise ValueError("No corpus available. Please provide a corpus.")
        
        # Determine search method
        search_with_faiss = self.use_faiss if use_faiss is None else use_faiss
        
        # Encode queries
        query_texts = list(queries.values())
        query_embeddings = self.model.encode(
            query_texts,
            convert_to_tensor=True,
            show_progress_bar=False
        )
        query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
        
        # Perform search based on method
        if search_with_faiss and self.faiss_index is not None:
            logger.info("Searching with FAISS...")
            query_embeddings_np = query_embeddings.cpu().numpy()
            top_k_values, top_k_indices = self._search_with_faiss(query_embeddings_np, top_k)
        else:
            logger.info("Searching with PyTorch...")
            top_k_values, top_k_indices = self._search_with_torch(query_embeddings, top_k)
        
        # Organize results
        results = {}
        for query_idx, (query_id, values, indices) in enumerate(zip(queries.keys(), top_k_values, top_k_indices)):
            query_results = {}
            for score, doc_idx in zip(values.cpu().numpy(), indices.cpu().numpy()):
                doc_id = self.corpus_ids[doc_idx]
                query_results[doc_id] = float(score)
            results[query_id] = query_results
        
        return results

def main():
    from utils.corpus_utils import load_or_process_corpus
    import time
    
    logger.info(f"Loading corpus from {corpus_path}...")
    corpus = load_or_process_corpus(corpus_path=corpus_path, pickle_path=pickle_path)
    
    # Test queries
    queries = {
        "q1": "large language models",
        "q2": "neural networks",
        "q3": "machine learning"
    }
    
    # Compare PyTorch and FAISS search
    logger.info("Creating retriever with PyTorch search...")
    retriever_torch = DenseRetriever(corpus=corpus, use_faiss=False)
    
    logger.info("Creating retriever with FAISS search...")
    retriever_faiss = DenseRetriever(corpus=corpus, use_faiss=True)
    
    # Test search speed with PyTorch
    logger.info("Testing search with PyTorch...")
    start_time = time.time()
    results_torch = retriever_torch.search(queries, top_k=10)
    torch_time = time.time() - start_time
    logger.info(f"PyTorch search took {torch_time:.4f} seconds")
    
    # Test search speed with FAISS
    logger.info("Testing search with FAISS...")
    start_time = time.time()
    results_faiss = retriever_faiss.search(queries, top_k=10)
    faiss_time = time.time() - start_time
    logger.info(f"FAISS search took {faiss_time:.4f} seconds")
    logger.info(f"Speedup: {torch_time/faiss_time:.2f}x")
    
    print("\nPyTorch Search Results:")
    for query_id, query_results in results_torch.items():
        print(f"\nQuery: {queries[query_id]}")
        for i, (doc_id, score) in enumerate(list(query_results.items())[:3]):
            doc = corpus[doc_id]
            print(f"Result {i+1}: {doc['title']} (Score: {score:.4f})")