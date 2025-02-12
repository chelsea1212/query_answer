import pandas as pd
import numpy as np
import streamlit as st
from openai import OpenAI
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from typing import List, Dict
import logging
import subprocess
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_csv():
    """
    Download the CSV file using gdown with fuzzy matching.
    Returns the path to the CSV file.
    """
    csv_path = "processed_data.csv"
    if not os.path.exists(csv_path):
        try:
            subprocess.run(
                ["gdown", "1W12c1_Jel_rodgIOVNBORO39C2cZ6uD5", "-O", "processed_data.csv", "--fuzzy"],
                check=True
            )
            logger.info(f"Successfully downloaded {csv_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error downloading file: {str(e)}")
            raise
    return csv_path

class EmailSearchEngine:
    def __init__(self, csv_path: str, api_key: str):
        """
        Initialize the email search engine.
        
        Args:
            csv_path (str): Path to the CSV file containing email data and embeddings
            api_key (str): OpenAI API key
        """
        self.client = OpenAI(api_key=api_key)
        self.df = pd.read_csv(csv_path)
        
        # Convert string representations of embeddings to numpy arrays
        self.df['embeddings'] = self.df['embeddings'].apply(
            lambda x: np.fromstring(x.strip('[]'), sep=',')
        )
        
        # Normalize embeddings for better cosine similarity calculation
        self.df['embeddings'] = self.df['embeddings'].apply(
            lambda x: x / np.linalg.norm(x)
        )
        
        # Initialize BM25 for keyword search
        self.bm25 = BM25Okapi(self.df['text'].apply(lambda x: x.split()).tolist())
        
        # Initialize cross-encoder for re-ranking
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        logger.info(f"Loaded {len(self.df)} emails from {csv_path}")

    def get_embedding(self, query: str) -> np.ndarray:
        """
        Get embeddings for a query using text-embedding-3-small model.
        
        Args:
            query (str): The search query
            
        Returns:
            np.ndarray: The embedding vector
        """
        try:
            response = self.client.embeddings.create(
                input=query,
                model="text-embedding-3-small"
            )
            embedding = np.array(response.data[0].embedding)
            embedding = embedding / np.linalg.norm(embedding)  # Normalize query embedding
            return embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {str(e)}")
            raise

    def expand_query(self, query: str) -> str:
        """
        Expand the query using GPT.
        
        Args:
            query (str): The original query
            
        Returns:
            str: Expanded query
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant. Generate synonyms or related terms for the query."
                    },
                    {
                        "role": "user",
                        "content": f"Generate synonyms or related terms for: {query}"
                    }
                ],
                max_tokens=50,
                temperature=0.7
            )
            expanded_query = response.choices[0].message.content
            return f"{query} {expanded_query}"
        except Exception as e:
            logger.error(f"Error expanding query: {str(e)}")
            return query  # Fallback to the original query

    def hybrid_search(self, query: str, top_k: int = 5, alpha: float = 0.5) -> pd.DataFrame:
        """
        Perform hybrid search (semantic + keyword).
        
        Args:
            query (str): The search query
            top_k (int): Number of top results to return
            alpha (float): Weight for semantic search (1 - alpha for keyword search)
            
        Returns:
            pd.DataFrame: Top k most similar emails with their combined scores
        """
        try:
            # Expand the query
            expanded_query = self.expand_query(query)
            
            # Semantic search
            query_embedding = self.get_embedding(expanded_query)
            embeddings_matrix = np.stack(self.df['embeddings'].values)
            semantic_scores = np.dot(embeddings_matrix, query_embedding)
            
            # Keyword search
            tokenized_query = expanded_query.split()
            keyword_scores = self.bm25.get_scores(tokenized_query)
            
            # Normalize scores
            semantic_scores = (semantic_scores - semantic_scores.min()) / (semantic_scores.max() - semantic_scores.min())
            keyword_scores = (keyword_scores - keyword_scores.min()) / (keyword_scores.max() - keyword_scores.min())
            
            # Combine scores
            combined_scores = alpha * semantic_scores + (1 - alpha) * keyword_scores
            
            # Add scores to the DataFrame
            self.df['combined_score'] = combined_scores
            initial_results = self.df.nlargest(top_k * 2, 'combined_score')  # Retrieve more results for re-ranking
            
            # Re-rank with cross-encoder
            pairs = [(expanded_query, doc) for doc in initial_results['text']]
            cross_encoder_scores = self.cross_encoder.predict(pairs)
            initial_results['cross_encoder_score'] = cross_encoder_scores
            
            # Return top-k re-ranked results
            return initial_results.nlargest(top_k, 'cross_encoder_score')[['text', 'cross_encoder_score']]
        except Exception as e:
            logger.error(f"Error during hybrid search: {str(e)}")
            raise

    def answer_query(self, query: str, max_tokens: int = 3800, 
                    temperature: float = 0.7) -> str:
        """
        Answer a query based on the most relevant emails.
        
        Args:
            query (str): The query to answer
            max_tokens (int): Maximum number of tokens for context
            temperature (float): Temperature for response generation
            
        Returns:
            str: Generated answer based on relevant emails
        """
        try:
            top_emails = self.hybrid_search(query)
            context = "\n\n".join(top_emails['text'])

            # Truncate context if it exceeds the maximum token length
            if len(context.split()) > max_tokens:
                context = " ".join(context.split()[:max_tokens])

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful email assistant. Provide clear, "
                            "concise, and detailed answers based on the email context provided. "
                            "Ensure your response addresses the query with precision and avoids ambiguity."
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Based on these emails:\n\n{context}\n\n"
                            f"Answer this query: {query}"
                        )
                    }
                ],
                max_tokens=1024,
                temperature=temperature
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            raise

# Streamlit app
def main():
    st.title("Email Search and Query Answering")

    # Download the CSV file if it doesn't exist
    try:
        with st.spinner("Downloading required data..."):
            csv_path = download_csv()
        st.success("Data loaded successfully!")
    except Exception as e:
        st.error(f"Error downloading data: {str(e)}")
        return

    # Use the provided API key and initialize the search engine
    api_key = "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"

    if api_key:
        try:
            search_engine = EmailSearchEngine(csv_path, api_key)

            # User input for the query
            query = st.text_input("Enter your query:")

            if query:
                with st.spinner("Searching for relevant emails and generating a response..."):
                    try:
                        answer = search_engine.answer_query(query)
                        st.subheader("Query Answer")
                        st.write(answer)
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        except Exception as e:
            st.error(f"Error initializing the search engine: {str(e)}")

if __name__ == "__main__":
    main()
