import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from retriever.dense_retriever import DenseRetriever
import logging
import re

logger = logging.getLogger(__name__)

class RAGModel:
    def __init__(self, retriever: DenseRetriever, model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"):
        self.retriever = retriever
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.generator = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True
        ).to(self.retriever.device)

    def extract_answer(self, full_response: str, query: str) -> str:
        """
        Extract only the answer part from the full response
        
        Args:
            full_response (str): The full response from the model
            query (str): The original query
            
        Returns:
            str: The extracted answer part
        """
        # Method 1: Extract content after "Answer:"
        if "Answer:" in full_response:
            answer = full_response.split("Answer:")[-1].strip()
            return answer
            
        # Method 2: Try to find where the generated answer begins after context
        if "Context:" in full_response:
            # Split by "Context:" and take everything after it
            after_context = full_response.split("Context:")[-1].strip()
            # Check if there's a clear separator, like a newline
            if "\n" in after_context:
                # Take the text after the first newline after "Context:"
                answer = after_context.split("\n", 1)[-1].strip()
                return answer
        
        # Method 3: Remove the input query if it appears at the beginning
        if full_response.startswith(f"Query: {query}"):
            without_query = full_response[len(f"Query: {query}"):].strip()
            # If there are recognizable sections, return text after the last recognized section
            sections = ["Context:", "Documents:", "Retrieved:"]
            for section in sections:
                if section in without_query:
                    answer = without_query.split(section)[-1].strip()
                    return answer
        
        # If all extraction methods fail, return the original response
        return full_response

    def generate(self, query: str, top_k: int = 5, max_new_tokens: int = 100, return_only_answer: bool = True) -> str:
        """
        Generate an answer for the query using the retrieved documents
        
        Args:
            query (str): The user query
            top_k (int): Number of documents to retrieve
            max_new_tokens (int): Maximum number of tokens to generate
            return_only_answer (bool): Whether to return only the answer part or the full response
            
        Returns:
            str: The generated answer (or full response if return_only_answer is False)
        """
        # Retrieve relevant documents
        query_dict = {"query": query}
        retrieved_docs = self.retriever.search(query_dict, top_k=top_k)["query"]
        # print(retrieved_docs)
        # import pdb; pdb.set_trace()
        
        # Prepare input for the generator
        context = " ".join([f"{doc_id}: {self.retriever.corpus[doc_id]['title']} {self.retriever.corpus[doc_id]['abstract']}" for doc_id in retrieved_docs])
        # Format prompt to explicitly include an Answer: section
        input_text = f"Query: {query}\nContext: {context}\nNow summarize the content and answer my question.\nAnswer:"
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.retriever.device)
        
        # Generate response
        output_ids = self.generator.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id
        )
        full_response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Return either the full response or just the answer part
        if return_only_answer:
            return self.extract_answer(full_response, query)
        else:
            return full_response

