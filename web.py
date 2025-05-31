import os
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import numpy as np
import chromadb
from chromadb.config import Settings
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import urllib3
import uuid
import logging

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
GEMINI_API_KEY = "AIzaSyBxB3DuQUG2t9EuGT3Ig4MGrQk6IbX0Ou4"  
CHROMA_DB_PATH = "./chroma_db"

# 🔒 Hardcoded website list
HARDCODED_URLS = [
    # HOME
    "https://3asolutions.co.in/home.aspx",
    # COMPANY
    "https://3asolutions.co.in/aboutus.aspx",
    "https://3asolutions.co.in/why3asolutions.aspx",
    # INDUSTRIES
    "https://3asolutions.co.in/industries.aspx",
    # PRODUCTS
    "https://3asolutions.co.in/erp.aspx",
    "https://3asolutions.co.in/hrandpayroll.aspx",
    "https://3asolutions.co.in/cleaningvalidation.aspx",
    "https://3asolutions.co.in/labapplication.aspx",
    "https://3asolutions.co.in/dms.aspx",
    "https://3asolutions.co.in/ocr.aspx",
    "https://3asolutions.co.in/rms.aspx",
    "https://3asolutions.co.in/ecomplaint.aspx",
    "https://3asolutions.co.in/eschedule.aspx",
    # SERVICES
    "https://3asolutions.co.in/softwaredevelopment.aspx",
    "https://3asolutions.co.in/softwaremaintenance.aspx",
]

# Initialize FastAPI app
app = FastAPI(
    title="Company Chatbot API",
    description="Multi-Website RAG Chatbot API using ChromaDB and Gemini",
    version="1.0.0"
)

# Pydantic models
class QuestionRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str

# Global variables
embedding_model = None
chroma_client = None
collection = None
gemini_model = None

def initialize_models():
    """Initialize all models and databases"""
    global embedding_model, chroma_client, collection, gemini_model
    
    try:
        # Initialize embedding model
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Embedding model initialized successfully")
        
        # Initialize Gemini
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        logger.info("Gemini model initialized successfully")
        
        # Initialize ChromaDB
        os.makedirs(CHROMA_DB_PATH, exist_ok=True)
        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        
        # Get or create collection
        collection_name = "company_knowledge_base"
        try:
            collection = chroma_client.get_collection(name=collection_name)
            logger.info(f"Existing ChromaDB collection '{collection_name}' loaded")
        except:
            collection = chroma_client.create_collection(name=collection_name)
            logger.info(f"New ChromaDB collection '{collection_name}' created")
            
            # Process and store website data
            process_and_store_websites()
            
    except Exception as e:
        logger.error(f"Error initializing models: {e}")
        raise

def extract_text_from_url(url: str) -> str:
    """Extract text content from a URL"""
    try:
        response = requests.get(url, timeout=10, verify=False)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        text = soup.get_text(separator="\n")
        # Clean up the text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text
    except Exception as e:
        logger.error(f"Error extracting text from {url}: {e}")
        return None

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks"""
    chunks = []
    words = text.split()
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if len(chunk.strip()) > 50:  # Only add meaningful chunks
            chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
            
    return chunks

def process_and_store_websites():
    """Process all websites and store in ChromaDB"""
    global collection, embedding_model
    
    logger.info("Starting website processing...")
    all_chunks = []
    all_metadatas = []
    all_ids = []
    
    for url in HARDCODED_URLS:
        logger.info(f"Processing: {url}")
        site_text = extract_text_from_url(url)
        
        if site_text:
            chunks = chunk_text(site_text)
            for i, chunk in enumerate(chunks):
                chunk_id = str(uuid.uuid4())
                all_chunks.append(chunk)
                all_metadatas.append({
                    "source_url": url,
                    "chunk_index": i,
                    "chunk_id": chunk_id
                })
                all_ids.append(chunk_id)
    
    if all_chunks:
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(all_chunks)} chunks...")
        embeddings = embedding_model.encode(all_chunks).tolist()
        
        # Store in ChromaDB
        collection.add(
            embeddings=embeddings,
            documents=all_chunks,
            metadatas=all_metadatas,
            ids=all_ids
        )
        
        logger.info(f"Successfully stored {len(all_chunks)} chunks in ChromaDB")
    else:
        logger.error("No chunks were created from the websites")

def retrieve_relevant_chunks(query: str, top_k: int = 3) -> List[str]:
    """Retrieve relevant chunks from ChromaDB"""
    global collection
    
    try:
        results = collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        return results['documents'][0] if results['documents'] else []
    except Exception as e:
        logger.error(f"Error retrieving chunks: {e}")
        return []

def generate_response(query: str, relevant_chunks: List[str]) -> str:
    """Generate response using Gemini"""
    global gemini_model
    
    context = "\n".join(relevant_chunks)
    prompt = f"""
    Based on the following website content, answer the user's question:

    Context:
    {context}

    Question: {query}

    Please provide a clear and concise answer based on the context provided.
    You are an AI assistant trained to provide helpful responses about 3A Solutions company.
    If asked about how to contact the company, generate the answer and give this website link: https://3asolutions.co.in/contactus.aspx
    If asked about careers in the company, generate the answer and give this website link: https://3asolutions.co.in/careers.aspx.
    If user asks you as the second person then decode it as the company and aner accordingly.
    For example: User: What are your company's objectives?
    In above example, 'your' referred as 3asolutions. Give the answer accordingly.

    If user asks you about how are you then respond that you are fine and how can I help you today.
    If user greets you, then greet back in a very polite way.
    If the context doesn't contain relevant information to answer the question, politely say that you don't have enough information about that specific topic in the available content.
    """
    
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return "I apologize, but I'm having trouble generating a response right now. Please try again later."

@app.on_event("startup")
async def startup_event():
    """Initialize models and database on startup"""
    logger.info("Starting up the application...")
    initialize_models()
    logger.info("Application startup completed successfully")

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Company Chatbot API is running", "version": "1.0.0"}

@app.post("/ask", response_model=ChatResponse)
async def ask_question(request: QuestionRequest):
    """Main endpoint to ask questions and get answers"""
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        logger.info(f"Received question: {request.question}")
        
        # Retrieve relevant chunks
        relevant_chunks = retrieve_relevant_chunks(request.question, top_k=3)
        
        if not relevant_chunks:
            return ChatResponse(
                answer="I apologize, but I couldn't find relevant information to answer your question. Please try rephrasing your question or ask about our company's products, services, or general information."
            )
        
        # Generate response
        answer = generate_response(request.question, relevant_chunks)
        
        logger.info("Response generated successfully")
        
        return ChatResponse(
            answer=answer
        )
        
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail="Internal server error occurred while processing your question")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
