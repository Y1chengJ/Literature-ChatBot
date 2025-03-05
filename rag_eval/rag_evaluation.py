import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import util
import numpy as np
import logging
from typing import Dict, List, Optional, Union
from model.rag_model import RAGModel
from utils.config_loader import load_from_config

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

class RAGEvaluator:
    """
    Evaluator for RAG system outputs
    """
    def __init__(self):
        # Initialize models for evaluation
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # For relevance evaluation
        logger.info("Loading relevance model...")
        self.relevance_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self.relevance_model = AutoModelForSequenceClassification.from_pretrained(self.relevance_model_name).to(self.device)
        self.relevance_tokenizer = AutoTokenizer.from_pretrained(self.relevance_model_name)

    def evaluate_relevance(self, query: str, answer: str) -> float:
        """
        Evaluate relevance of the answer to the query
        
        Args:
            query (str): User query
            answer (str): Generated answer
            
        Returns:
            float: Relevance score between 0 and 1
        """
        # Use cross-encoder to evaluate relevance
        inputs = self.relevance_tokenizer(query, answer, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        
        with torch.no_grad():
            scores = self.relevance_model(**inputs).logits
        
        score = torch.sigmoid(scores[0]).cpu().numpy()[0]
        return float(score)
    
    def evaluate_context_coverage(self, answer: str, contexts: List[str]) -> float:
        """
        Evaluate how well the answer covers the information in retrieved contexts
        
        Args:
            answer (str): Generated answer
            contexts (List[str]): Retrieved context documents
            
        Returns:
            float: Coverage score between 0 and 1
        """
        if not contexts:
            return 0.0
            
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        
        answer_embedding = model.encode([answer], convert_to_tensor=True)
        context_embeddings = model.encode(contexts, convert_to_tensor=True)
        
        # Calculate cosine similarities
        similarities = util.pytorch_cos_sim(answer_embedding, context_embeddings)[0]
        
        # Take average of top 3 similarities or all if less than 3
        k = min(3, len(contexts))
        top_similarities, _ = torch.topk(similarities, k=k)
        
        return float(torch.mean(top_similarities).cpu().numpy())
    
    def evaluate_factual_consistency(self, answer: str, contexts: List[str]) -> float:
        """
        Evaluate the factual consistency of the answer with retrieved contexts
        
        Args:
            answer (str): Generated answer
            contexts (List[str]): Retrieved context documents
            
        Returns:
            float: Consistency score between 0 and 1
        """
        answer_terms = set(answer.lower().split())
        
        # Remove common words
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 
                     'at', 'to', 'for', 'with', 'by', 'about', 'as', 
                     'of', 'that', 'this', 'is', 'are', 'was', 'were'}
        answer_terms -= stopwords
        
        if not answer_terms:
            return 0.5  # Neutral score if no meaningful terms in answer
        
        # Count how many terms from the answer appear in contexts
        context_text = " ".join(contexts).lower()
        matching_terms = sum(1 for term in answer_terms if term in context_text)
        
        # Calculate overlap ratio
        overlap_ratio = matching_terms / len(answer_terms)
        
        return overlap_ratio

    def evaluate_answer_quality(self, answer: str) -> float:
        """
        Evaluate the linguistic quality of the answer
        
        Args:
            answer (str): Generated answer
            
        Returns:
            float: Quality score between 0 and 1
        """
        # Length check - too short answers are penalized
        length_score = min(1.0, len(answer) / 200)
        
        # Structure check - sentences should end with proper punctuation
        sentences = answer.split('. ')
        proper_endings = sum(1 for s in sentences if s.strip() and (s.strip()[-1] in '.!?:;'))
        structure_score = proper_endings / max(1, len(sentences))
        
        # Calculate average score
        quality_score = (length_score + structure_score) / 2
        
        return quality_score

    def evaluate_rag_answer(self, query: str, answer: str, retrieved_docs: Dict[str, Dict], 
                           detailed: bool = False) -> Union[float, Dict[str, float]]:
        """
        Perform comprehensive evaluation of a RAG-generated answer
        
        Args:
            query (str): User query
            answer (str): Generated answer
            retrieved_docs (Dict): Retrieved documents used for generation
            detailed (bool): Whether to return detailed scores or overall score
            
        Returns:
            Union[float, Dict[str, float]]: Overall score or detailed scores
        """
        # Extract contexts from retrieved documents
        contexts = []
        for doc_id, doc_info in retrieved_docs.items():
            if isinstance(doc_info, dict) and 'title' in doc_info and 'abstract' in doc_info:
                contexts.append(f"{doc_info['title']} {doc_info['abstract']}")
            elif isinstance(doc_info, float):  # It's just a score
                pass
        
        # Calculate individual scores
        relevance_score = self.evaluate_relevance(query, answer)
        coverage_score = self.evaluate_context_coverage(answer, contexts)
        consistency_score = self.evaluate_factual_consistency(answer, contexts)
        quality_score = self.evaluate_answer_quality(answer)
        
        # Calculate overall score with weighted average
        weights = {
            'relevance': 0.35,
            'coverage': 0.25,
            'consistency': 0.25,
            'quality': 0.15
        }
        
        overall_score = (
            weights['relevance'] * relevance_score +
            weights['coverage'] * coverage_score +
            weights['consistency'] * consistency_score +
            weights['quality'] * quality_score
        )
        
        if detailed:
            return {
                'overall': overall_score,
                'relevance': relevance_score,
                'coverage': coverage_score,
                'consistency': consistency_score,
                'quality': quality_score
            }
        else:
            return overall_score

    def get_evaluation_explanation(self, scores: Dict[str, float]) -> str:
        """
        Generate a human-readable explanation of the evaluation scores
        
        Args:
            scores (Dict[str, float]): Evaluation scores
            
        Returns:
            str: Explanation text
        """
        explanations = []
        
        # Overall score explanation
        overall = scores['overall']
        if overall >= 0.8:
            explanations.append("Excellent answer overall.")
        elif overall >= 0.6:
            explanations.append("Good answer with some minor issues.")
        elif overall >= 0.4:
            explanations.append("Acceptable answer but has significant room for improvement.")
        else:
            explanations.append("Poor answer, needs substantial improvement.")
        
        # Relevance explanation
        relevance = scores['relevance']
        if relevance >= 0.8:
            explanations.append("- Highly relevant to the query.")
        elif relevance >= 0.6:
            explanations.append("- Mostly relevant to the query.")
        elif relevance >= 0.4:
            explanations.append("- Somewhat relevant, but could address the query better.")
        else:
            explanations.append("- Not very relevant to the query.")
        
        # Coverage explanation
        coverage = scores['coverage']
        if coverage >= 0.8:
            explanations.append("- Excellent coverage of the retrieved information.")
        elif coverage >= 0.6:
            explanations.append("- Good coverage of important information from sources.")
        elif coverage >= 0.4:
            explanations.append("- Partial coverage of source information.")
        else:
            explanations.append("- Poor coverage of information from sources.")
        
        # Consistency explanation
        consistency = scores['consistency']
        if consistency >= 0.8:
            explanations.append("- Very consistent with source facts.")
        elif consistency >= 0.6:
            explanations.append("- Generally consistent with source facts.")
        elif consistency >= 0.4:
            explanations.append("- Some inconsistencies with source facts.")
        else:
            explanations.append("- Many inconsistencies with source information.")
        
        # Quality explanation
        quality = scores['quality']
        if quality >= 0.8:
            explanations.append("- Well-structured and comprehensive.")
        elif quality >= 0.6:
            explanations.append("- Adequately structured and mostly complete.")
        elif quality >= 0.4:
            explanations.append("- Basic structure but lacking detail.")
        else:
            explanations.append("- Poorly structured or incomplete.")
            
        return "\n".join(explanations)


def evaluate_rag_system(rag_model: RAGModel, queries: List[str], top_k: int = 5, 
                       max_new_tokens: int = 200) -> Dict:
    """
    Evaluate a RAG system over a set of queries
    
    Args:
        rag_model (RAGModel): The RAG model to evaluate
        queries (List[str]): List of test queries
        top_k (int): Number of documents to retrieve
        max_new_tokens (int): Maximum number of tokens to generate
        
    Returns:
        Dict: Evaluation results
    """
    evaluator = RAGEvaluator()
    results = {
        'queries': [],
        'answers': [],
        'scores': [],
        'detailed_scores': []
    }
    
    for query in queries:
        logger.info(f"Processing query: {query}")
        
        # Get retrieved documents
        query_dict = {"query": query}
        retrieved_docs = rag_model.retriever.search(query_dict, top_k=top_k)["query"]
        
        # Generate answer - ensure we only get the answer part
        answer = rag_model.generate(query, top_k=top_k, max_new_tokens=max_new_tokens, return_only_answer=True)
        
        # Evaluate answer
        detailed_scores = evaluator.evaluate_rag_answer(
            query=query, 
            answer=answer, 
            retrieved_docs={doc_id: rag_model.retriever.corpus[doc_id] for doc_id in retrieved_docs},
            detailed=True
        )
        
        # Save results
        results['queries'].append(query)
        results['answers'].append(answer)
        results['scores'].append(detailed_scores['overall'])
        results['detailed_scores'].append(detailed_scores)
        
        # Log scores
        logger.info(f"Overall score: {detailed_scores['overall']:.4f}")
        logger.info(f"Relevance: {detailed_scores['relevance']:.4f}, " +
                   f"Coverage: {detailed_scores['coverage']:.4f}, " +
                   f"Consistency: {detailed_scores['consistency']:.4f}, " +
                   f"Quality: {detailed_scores['quality']:.4f}")
    
    # Calculate average scores
    results['avg_overall'] = np.mean(results['scores'])
    avg_detailed = {
        'relevance': np.mean([s['relevance'] for s in results['detailed_scores']]),
        'coverage': np.mean([s['coverage'] for s in results['detailed_scores']]),
        'consistency': np.mean([s['consistency'] for s in results['detailed_scores']]),
        'quality': np.mean([s['quality'] for s in results['detailed_scores']])
    }
    results['avg_detailed'] = avg_detailed
    
    return results


# Test the evaluator
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Load corpus and model
    from utils.corpus_utils import load_or_process_corpus
    corpus = load_or_process_corpus(corpus_path=corpus_path, pickle_path=pickle_path)
    
    # Set up retriever and RAG model
    from retriever.dense_retriever import DenseRetriever
    retriever = DenseRetriever(corpus=corpus, use_faiss=True)
    
    from model.rag_model import RAGModel
    rag_model = RAGModel(retriever)
    
    # Test queries
    test_queries = [
        "Explain how transformer models work in NLP",
        "What are the limitations of large language models?",
        "How does uncertainty quantification improve model reliability?"
    ]
    
    # Evaluate RAG system
    results = evaluate_rag_system(rag_model, test_queries)
    
    # Print summary results
    print("\nEvaluation Summary:")
    print(f"Average Overall Score: {results['avg_overall']:.4f}")
    print(f"Average Relevance: {results['avg_detailed']['relevance']:.4f}")
    print(f"Average Coverage: {results['avg_detailed']['coverage']:.4f}")
    print(f"Average Consistency: {results['avg_detailed']['consistency']:.4f}")
    print(f"Average Quality: {results['avg_detailed']['quality']:.4f}")
    
    # Print individual results
    print("\nIndividual Results:")
    for i, (query, answer, score) in enumerate(zip(results['queries'], results['answers'], results['detailed_scores'])):
        print(f"\nQuery {i+1}: {query}")
        print(f"Answer: {answer[:100]}...")
        print(f"Overall Score: {score['overall']:.4f}")
