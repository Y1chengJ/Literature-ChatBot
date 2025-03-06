# Literature Chatbot

This project implements a Retrieval-Augmented Generation (RAG) system for academic literature. It uses dense retrievers with FAISS indexing for efficient document retrieval and language models for answer generation. The data provided spans over five categories—Machine Learning, Artificial Intelligence, Computer Vision, Statistical Machine Learning, and Natural Language Processing—with publication dates ranging from April 2022 to the present. 

## Project Structure

- **data/**: Data handling utilities
  - ```unzip data.zip``` to extract the data
- **embeddings/**: Storage for pre-computed embeddings
- **models/**: Model implementations
- **retrievers/**: Retrieval components (dense retrievers, FAISS retrievers)
- **utils/**: Utility functions
- **src/app.py**: Web application using Gradio
- **config.toml**: configs that can be changed based on preference

## Setup and Installation

Create the Env

```python
conda create -n LitChatbot python==3.10
pip install -r requirements.txt
conda activate LitChatbot
```

Run the Gradio Interface

```python
python src/app.py
```

This will launch a Gradio web interface where you can interact with the RAG system.

## Features

- **Academic Query Answering**: Ask questions about academic literature and get AI-generated answers
- **Document Retrieval**: View the source documents used to generate answers
- **Feedback System**: Rate answers and provide detailed feedback
- **Configurable Search**: Adjust search parameters like number of documents to retrieve
- **FAISS Acceleration**: Option to use FAISS for faster search on large datasets

## Data Pipeline

If you want to crawl and use your own data, follow the following process:

1. Crawling 

   ```python	
   python arxiv_crawler/arxiv_crawler.py
   ```

2. Corpus building 

   ```python
   python arxiv_crawler/build_corpus.py
   ```

3. Embedding generation

   When you first run `src\app.py`, it will generate embedding for you, if the embedding does not exist under data folder.

## Model Architecture

The RAG architecture consists of:
1. **Retriever Component**: Dense retrievers using embeddings to find relevant documents
2. **Generator Component**: Language model that synthesizes answers from retrieved content

To test each component:

module_name can be `rag`, `retriever`, `all`

```python
python test/test.py --test module_name
```

## Evaluation

The system includes an evaluation framework in to measure the quality of generated answers.

```python
python rag_eval/rag_evaluation.py
```

# Config

Feel free to change the LLM model and the embedding model based in your needs.

In default, the used embedding is encoded by `intfloat/multilingual-e5-large-instruct`, but I also provide another embedding created by ` BAAI/bge-m3`. 
