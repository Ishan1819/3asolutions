# import requests
# from bs4 import BeautifulSoup
# import chromadb
# from chromadb.utils import embedding_functions
# import google.generativeai as genai
# import os
# from typing import Dict, Any

# # -------------------------------
# # Configuration
# # -------------------------------
# GEMINI_API_KEY = "AIzaSyCesYn4sYjm8ZH2cTpEc6pgsOM8MNqeOfU"
# genai.configure(api_key=GEMINI_API_KEY)

# # -------------------------------
# # Embedding & Chroma Setup
# # -------------------------------
# model_name = "all-MiniLM-L6-v2"
# embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)

# chroma_client = chromadb.Client()
# collection = chroma_client.get_or_create_collection("web_content", embedding_function=embedding_func)

# # -------------------------------
# # Tool Functions
# # -------------------------------
# def retrieve_url_tool(url: str, query: str = None) -> str:
#     print(f"[TOOL] Retrieving URL: {url}")
#     try:
#         response = requests.get(url, timeout=10)
#         response.raise_for_status()
#         soup = BeautifulSoup(response.text, "html.parser")
        
#         for script in soup(["script", "style"]):
#             script.decompose()
        
#         paragraphs = [p.get_text() for p in soup.find_all("p")]
#         chunks = [p.strip() for p in paragraphs if len(p.strip()) > 100]

#         try:
#             existing_ids = [f"{url}_{i}" for i in range(len(chunks))]
#             collection.delete(ids=existing_ids)
#         except:
#             pass
        
#         for i, chunk in enumerate(chunks):
#             collection.add(documents=[chunk], ids=[f"{url}_{i}"])

#         # If query is provided, directly answer
#         if query:
#             results = collection.query(query_texts=[query], n_results=5)
#             if not results["documents"][0]:
#                 return "✅ Content retrieved but no relevant results found for your query."

#             combined_text = " ".join(results["documents"][0])
#             model = genai.GenerativeModel("gemini-1.5-flash")
#             response = model.generate_content(f"Answer this based on the retrieved web content: {query}\n\n{combined_text}")
#             return response.text.strip()
        
#         return f"✅ Retrieved and stored {len(chunks)} chunks from {url}"
    
#     except Exception as e:
#         return f"❌ Error retrieving URL: {str(e)}"


# def summarize_content_tool(query: str) -> str:
#     print(f"[TOOL] Summarizing content for query: {query}")
#     try:
#         results = collection.query(query_texts=[query], n_results=5)
        
#         if not results["documents"][0]:
#             return "❌ No content found to summarize. Please retrieve content first."
        
#         combined_text = " ".join(results["documents"][0])
        
#         model = genai.GenerativeModel("gemini-1.5-flash")
#         response = model.generate_content(f"Summarize the following content about {query}:\n\n{combined_text}")
        
#         return response.text.strip()
#     except Exception as e:
#         return f"❌ Error summarizing content: {str(e)}"

# # -------------------------------
# # RAG Agent Class (Gemini-based)
# # -------------------------------
# class RAGAgent:
#     def __init__(self, gemini_api_key: str):
#         genai.configure(api_key=gemini_api_key)
#         self.model = genai.GenerativeModel("gemini-1.5-flash")
#         self.context = ""
    
#     def retrieve_content(self, url: str, query: str = None) -> str:
#         result = retrieve_url_tool(url, query)
#         self.context = f"Content retrieved from {url}"
#         return result

    
#     def summarize_content(self, query: str) -> str:
#         return summarize_content_tool(query)
    
#     def chat(self, message: str) -> str:
#         try:
#             prompt = f"""You are a RAG (Retrieval-Augmented Generation) assistant.

# Context: {self.context}

# User message: {message}

# Based on the user's message, you need to:
# 1. If they want to retrieve content from a URL, respond with: "RETRIEVE: [URL]"
# 2. If they want to summarize content, respond with: "SUMMARIZE: [query]"
# 3. Otherwise, provide a helpful response about what you can do.

# Your capabilities:
# - Retrieve content from web URLs
# - Summarize retrieved content
# - Answer questions about AI, technology, etc.

# Please respond appropriately to: {message}
# """
#             response = self.model.generate_content(prompt)
#             return response.text.strip()
#         except Exception as e:
#             return f"Error processing request: {str(e)}"

# # -------------------------------
# # Simple RAG Implementation
# # -------------------------------
# def simple_rag_workflow(url: str, query: str = "artificial intelligence") -> str:
#     print("=== Simple RAG Workflow ===")
#     print("Step 1: Retrieving content...")
#     retrieval_result = retrieve_url_tool(url)
#     print(retrieval_result)

#     print("Step 2: Summarizing content...")
#     summary_result = summarize_content_tool(query)
#     print("Summary:")
#     print(summary_result)

#     return summary_result

# def interactive_rag_chat():
#     print("=== RAG Multi-Agent Chatbot (Gemini-powered) ===")
#     print("Commands:")
#     print("- Type a URL to retrieve content")
#     print("- Ask questions to get summaries")
#     print("- Type 'quit' to exit\n")

#     agent = RAGAgent(GEMINI_API_KEY)

#     while True:
#         user_input = input("You: ").strip()

#         if user_input.lower() in ['quit', 'exit', 'q']:
#             print("Goodbye!")
#             break

#         if not user_input:
#             continue

#         if user_input.startswith('http'):
#             print("Retrieving content...")
#             result = agent.retrieve_content(user_input)
#             print(f"Agent: {result}")
#             continue

#         response = agent.chat(user_input)

#         if response.startswith("RETRIEVE:"):
#             parts = response.replace("RETRIEVE:", "").strip().split("|")
#             url = parts[0].strip()
#             query = parts[1].strip() if len(parts) > 1 else None
#             result = agent.retrieve_content(url, query)
#             print(f"Agent: {result}")
#         elif response.startswith("SUMMARIZE:"):
#             query = response.replace("SUMMARIZE:", "").strip()
#             result = agent.summarize_content(query)
#             print(f"Agent: {result}")
#         else:
#             print(f"Agent: {response}")
# if __name__ == "__main__":
#     # Example usage
#     url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
#     query = "What is artificial intelligence?"
    
#     # Run the simple RAG workflow
#     simple_rag_workflow(url, query)
    
#     # Start the interactive chat
#     interactive_rag_chat()



import requests
from bs4 import BeautifulSoup
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai
import os
from typing import Dict, Any

# -------------------------------
# Configuration
# -------------------------------
GEMINI_API_KEY = "AIzaSyCesYn4sYjm8ZH2cTpEc6pgsOM8MNqeOfU"
genai.configure(api_key=GEMINI_API_KEY)

# -------------------------------
# Embedding & Chroma Setup
# -------------------------------
model_name = "all-MiniLM-L6-v2"
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)

chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection("web_content", embedding_function=embedding_func)

# -------------------------------
# Tool Functions
# -------------------------------
def retrieve_url_tool(url: str) -> str:
    print(f"[TOOL] Retrieving URL: {url}")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        for script in soup(["script", "style"]):
            script.decompose()
        
        paragraphs = [p.get_text() for p in soup.find_all("p")]
        chunks = [p.strip() for p in paragraphs if len(p.strip()) > 100]

        # Delete existing chunks for the URL to avoid duplicates
        try:
            existing_ids = [f"{url}_{i}" for i in range(len(chunks))]
            collection.delete(ids=existing_ids)
        except Exception:
            pass
        
        # Add new chunks to the collection
        for i, chunk in enumerate(chunks):
            collection.add(documents=[chunk], ids=[f"{url}_{i}"])

        return f"✅ Retrieved and stored {len(chunks)} chunks from {url}"
    
    except Exception as e:
        return f"❌ Error retrieving URL: {str(e)}"


def answer_query_tool(query: str) -> str:
    print(f"[TOOL] Answering query: {query}")
    try:
        results = collection.query(query_texts=[query], n_results=5)
        
        if not results["documents"][0]:
            return "❌ No relevant content found. Please retrieve the data first."
        
        combined_text = " ".join(results["documents"][0])
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = (
            f"Answer the following question based on the provided content:\n\n"
            f"Content: {combined_text}\n\n"
            f"Question: {query}\n\nAnswer:"
        )
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"❌ Error answering query: {str(e)}"


def summarize_content_tool(query: str) -> str:
    print(f"[TOOL] Summarizing content for query: {query}")
    try:
        results = collection.query(query_texts=[query], n_results=5)
        
        if not results["documents"][0]:
            return "❌ No content found to summarize. Please retrieve content first."
        
        combined_text = " ".join(results["documents"][0])
        
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"Summarize the following content about {query}:\n\n{combined_text}"
        response = model.generate_content(prompt)
        
        return response.text.strip()
    except Exception as e:
        return f"❌ Error summarizing content: {str(e)}"

# -------------------------------
# RAG Agent Class (Gemini-based)
# -------------------------------
class RAGAgent:
    def __init__(self, gemini_api_key: str):
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")
        self.context = ""  # Could store last retrieved URL or summary text if needed
    
    def retrieve_content(self, url: str) -> str:
        result = retrieve_url_tool(url)
        self.context = f"Content retrieved from {url}"
        return result

    def answer_query(self, query: str) -> str:
        return answer_query_tool(query)

    def summarize_content(self, query: str) -> str:
        return summarize_content_tool(query)
    
    def chat(self, message: str) -> str:
        try:
            prompt = f"""You are a RAG (Retrieval-Augmented Generation) assistant.

        You can do the following based on user input:
        - If the user inputs a URL, respond with: "RETRIEVE: [URL]"
        - If the user wants a summary, respond with: "SUMMARIZE: [query]"
        - If the user asks a question, answer it using the stored content

        User message: {message}

        Respond with one of the commands (RETRIEVE, SUMMARIZE) or answer the question directly.
"""
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Error processing request: {str(e)}"

# -------------------------------
# Simple RAG Workflow (for quick test)
# -------------------------------
def simple_rag_workflow(url: str, query: str = "artificial intelligence") -> str:
    print("=== Simple RAG Workflow ===")
    print("Step 1: Retrieving content...")
    retrieval_result = retrieve_url_tool(url)
    print(retrieval_result)

    print("Step 2: Answering query...")
    answer_result = answer_query_tool(query)
    print("Answer:")
    print(answer_result)

    return answer_result

# -------------------------------
# Interactive RAG Chatbot
# -------------------------------
def interactive_rag_chat():
    print("=== RAG Multi-Agent Chatbot (Gemini-powered) ===")
    print("Instructions:")
    print("- Enter a URL to load content.")
    print("- Ask questions related to the loaded content.")
    print("- Type 'quit' to exit.\n")

    agent = RAGAgent(GEMINI_API_KEY)

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if not user_input:
            continue

        # If input looks like a URL, retrieve and store content
        if user_input.startswith("http"):
            print("Retrieving content...")
            result = agent.retrieve_content(user_input)
            print(f"Agent: {result}")
            continue

        # Otherwise treat input as a query - answer from stored content
        response = agent.answer_query(user_input)
        print(f"Agent: {response}")


if __name__ == "__main__":
    # Example quick test
    url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
    query = "What is artificial intelligence?"
    
    simple_rag_workflow(url, query)
    
    # Start interactive chatbot
    interactive_rag_chat()
