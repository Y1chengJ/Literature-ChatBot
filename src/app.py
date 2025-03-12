import gradio as gr
import logging
import torch
from retriever.dense_retriever import DenseRetriever
from model.rag_model import RAGModel
from utils.corpus_utils import load_or_process_corpus
from utils.config_loader import load_from_config
import os
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
config = load_from_config()

# Load paths config
paths_config = config['paths']
corpus_path = paths_config['corpus_path'] if os.path.exists(paths_config['corpus_path']) else None
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


# Load RAG model
def load_rag_model(corpus_path=corpus_path, embeddings_dir=embeddings_dir, 
                  pickle_path=pickle_path, use_faiss=True):
    # Load or process corpus
    corpus = load_or_process_corpus(corpus_path, pickle_path)
    
    # Set up retriever and RAG model - auto process embeddings
    logger.info("Setting up retriever and RAG model...")
    retriever = DenseRetriever(
        corpus=corpus,
        model_name=retriever_model_name,
        embeddings_dir=embeddings_dir,
        use_faiss=use_faiss,
        batch_size=retriever_bath_size
    )
    
    rag_model = RAGModel(retriever, model_name=rag_model_name)
    return rag_model

# Global variables to track conversation history
chat_history = []
retrieved_docs_history = {}
current_query = ""

# Process user input function
def process_query(query, history, top_k=5, max_new_tokens=200, use_faiss=True):
    global current_query, retrieved_docs_history
    current_query = query
    
    logger.info(f"Processing query: {query}")
    
    # Retrieve relevant documents - explicitly specify whether to use FAISS
    query_dict = {"query": query}
    retrieved_docs = rag_model.retriever.search(query_dict, top_k=top_k, use_faiss=use_faiss)["query"]
    
    # Record retrieved documents
    doc_info = []
    for doc_id, score in retrieved_docs.items():
        doc = rag_model.retriever.corpus[doc_id]
        doc_info.append({
            "doc_id": doc_id,
            "title": doc['title'],
            "abstract": doc['abstract'][:200] + "..." if len(doc['abstract']) > 200 else doc['abstract'],
            "score": score
        })
    
    retrieved_docs_history[query] = doc_info
    
    answer = rag_model.generate(query, top_k=top_k, max_new_tokens=max_new_tokens, return_only_answer=True)
    
    # Update conversation history
    history.append((query, answer))
    
    return history, get_retrieved_docs_html(query)

# Generate HTML for retrieved documents
def get_retrieved_docs_html(query=None):
    if not query:
        query = current_query
        
    if query not in retrieved_docs_history:
        return "No retrieval records"
    
    docs = retrieved_docs_history[query]
    
    html = "<h3>Retrieved Documents:</h3>"
    for i, doc in enumerate(docs, 1):
        html += f"<div style='margin-bottom: 15px; padding: 10px; border: 1px solid #ddd; border-radius: 5px;'>"
        html += f"<p><strong>Document {i}:</strong> {doc['title']}</p>"
        html += f"<p><strong>Abstract:</strong> {doc['abstract']}</p>"
        html += f"<p><strong>Relevance Score:</strong> {doc['score']:.4f}</p>"
        html += "</div>"
    
    return html

# Save user feedback function
def save_feedback(score, feedback_text):
    if current_query:
        feedback_file = "user_feedback.txt"
        with open(feedback_file, "a", encoding="utf-8") as f:
            f.write(f"Query: {current_query}\n")
            f.write(f"Score: {score}\n")
            f.write(f"Feedback: {feedback_text}\n")
            f.write("-" * 50 + "\n")
        return f"Thank you for your feedback! Rating: {score}, feedback saved."
    return "No current query record, cannot save feedback."

# Load model
logger.info("Initializing RAG model...")
rag_model = load_rag_model()
logger.info("RAG model initialized successfully!")

# Create Gradio interface
with gr.Blocks(title="Academic RAG Dialogue System", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Academic Literature RAG Dialogue System")
    gr.Markdown("This system uses academic literature as a knowledge base to answer your academic questions.")
    
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=500)
            with gr.Row():
                msg = gr.Textbox(
                    show_label=False, 
                    placeholder="Enter your question...", 
                    container=False,
                    scale=8
                )
                submit_btn = gr.Button("Send", scale=1)
            
            with gr.Accordion("Provide Feedback", open=False):
                with gr.Row():
                    feedback_score = gr.Slider(minimum=1, maximum=5, step=1, label="Answer Quality Rating")
                    feedback_text = gr.Textbox(label="Detailed Feedback", placeholder="Please enter your detailed feedback...")
                feedback_btn = gr.Button("Submit Feedback")
                feedback_result = gr.Textbox(label="Feedback Result", interactive=False)
            
            with gr.Accordion("Search Settings", open=False):
                with gr.Row():
                    use_faiss = gr.Checkbox(value=True, label="Use FAISS for search", 
                                          info="Turn on for faster search on large datasets")
                    top_k = gr.Slider(minimum=1, maximum=20, value=5, step=1, label="Number of documents to retrieve")
        
        with gr.Column(scale=1):
            retrieved_docs = gr.HTML(label="Retrieved Documents")
    
    # Set up event handlers with new parameters
    submit_btn.click(process_query, 
                    inputs=[msg, chatbot, top_k, gr.Number(value=200, visible=False), use_faiss], 
                    outputs=[chatbot, retrieved_docs])
    msg.submit(process_query, 
              inputs=[msg, chatbot, top_k, gr.Number(value=200, visible=False), use_faiss], 
              outputs=[chatbot, retrieved_docs])
    feedback_btn.click(save_feedback, inputs=[feedback_score, feedback_text], outputs=feedback_result)
    
    # Example questions
    gr.Examples(
        examples=[
            "What are large language models? What are their applications?",
            "Explain uncertainty quantification in natural language generation.",
            "What are the latest methods in conformal prediction?",
            "How are deep learning models applied in scientific fields?"
        ],
        inputs=msg
    )
    
    gr.Markdown("## Instructions")
    gr.Markdown("1. Enter your academic question in the text box")
    gr.Markdown("2. The system will retrieve relevant literature and generate an answer")
    gr.Markdown("3. You can view the retrieved relevant documents on the right")
    gr.Markdown("4. If you wish, you can provide feedback on the quality of the answer")

# Launch the application
if __name__ == "__main__":
    demo.launch(share=True)
