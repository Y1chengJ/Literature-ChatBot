import logging
import argparse
from retriever.dense_retriever import DenseRetriever
from utils.corpus_utils import load_or_process_corpus
from utils.config_loader import load_from_config
from model.rag_model import RAGModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
config = load_from_config()

# Load paths config
paths_config = config['paths']
corpus_path = paths_config['corpus_path']
pickle_path = paths_config['pickle_path']
embeddings_dir = paths_config['embeddings_dir']

# Load retriever config
retriever_config = config['retriever']
use_faiss = retriever_config['use_faiss']
retriever_model_name = retriever_config['model_name']
retriever_bath_size = retriever_config['batch_size']

# Load RAG config
rag_config = config['rag']
rag_model_name = rag_config['model_name']
max_new_tokens = rag_config['max_new_tokens']
top_k = rag_config['top_k']


def test_retriever_speed():
    logging.basicConfig(level=logging.INFO)
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
            
            
def test_rag_model():
    logging.basicConfig(level=logging.INFO)
    
    # Load from processed corpus
    from utils.corpus_utils import load_or_process_corpus
    
    logger.info("Loading corpus...")
    corpus = load_or_process_corpus(corpus_path=corpus_path, pickle_path=pickle_path)
    
    # Load the retriever
    retriever = DenseRetriever(
        model_name=retriever_model_name,
        batch_size=retriever_bath_size,
        corpus=corpus,
        embeddings_dir=embeddings_dir,
        use_faiss=use_faiss
    )
    # print(retriever)
    # import pdb; pdb.set_trace() 
    if not retriever.load_vectors():
        raise ValueError("Corpus vectors not found. Please encode the corpus first.")
    
    # Create RAG model
    rag_model = RAGModel(
        model_name=rag_model_name,
        retriever=retriever
    )
    
    # Test queries
    test_queries = [
        "What methods are used for uncertainty quantification in natural language generation?",
        "Explain how large language models work and their applications.",
        "What are the latest advances in conformal prediction?"
    ]
    
    # Generate answers for test queries
    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}: {query}")
        logger.info(f"Generating answer for query {i}...")
        
        # Get both full response and answer-only
        answer_only = rag_model.generate(
            query,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            return_only_answer=True
        )
        
        print("\nAnswer Only:")
        print(answer_only)
        print("-" * 80)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run tests for Literature Chatbot')
    parser.add_argument('--test', type=str, default='all',
                      choices=['retriever', 'rag', 'all'],
                      help='Which test to run: retriever, rag, or all (default: all)')
    parser.add_argument('--top-k', type=int, default=None,
                      help=f'Number of documents to retrieve (default: {top_k} from config)')
    parser.add_argument('--max-tokens', type=int, default=None,
                      help=f'Maximum new tokens for generation (default: {max_new_tokens} from config)')
    parser.add_argument('--use-faiss', action='store_true', default=None,
                      help=f'Use FAISS for retrieval (default: {use_faiss} from config)')
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    # Override config with command-line arguments if provided
    global top_k, max_new_tokens, use_faiss
    if args.top_k is not None:
        top_k = args.top_k
    if args.max_tokens is not None:
        max_new_tokens = args.max_tokens
    if args.use_faiss is not None:
        use_faiss = args.use_faiss
    
    # Run the selected tests
    if args.test in ['retriever', 'all']:
        print("\n" + "="*50)
        print("Running retriever speed test...")
        print("="*50)
        test_retriever_speed()
    
    if args.test in ['rag', 'all']:
        print("\n" + "="*50)
        print("Running RAG model test...")
        print("="*50)
        test_rag_model()


if __name__ == "__main__":
    main()