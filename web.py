# import streamlit as st
# import speech_recognition as sr
# import pyttsx3
# import requests
# from bs4 import BeautifulSoup
# import google.generativeai as genai
# from sentence_transformers import SentenceTransformer
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# import tempfile
# import os
# from concurrent.futures import ThreadPoolExecutor
# import time
# import urllib3

# # Disable SSL warnings
# urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# @st.cache_resource
# def init_models():
#     recognizer = sr.Recognizer()
#     microphone = sr.Microphone()
#     embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
#     return recognizer, microphone, embedding_model

# def setup_gemini(api_key):
#     genai.configure(api_key=api_key)
#     return genai.GenerativeModel('gemini-1.5-flash')

# def extract_text_from_url(url):
#     try:
#         response = requests.get(url, timeout=10, verify=False)  
#         response.raise_for_status()
#         soup = BeautifulSoup(response.content, "html.parser")
#         text = soup.get_text(separator="\n")
#         return text
#     except Exception as e:
#         st.error(f"Error extracting website content: {e}")
#         return None

# def chunk_text(text, chunk_size=500, overlap=50):
#     chunks = []
#     words = text.split()
#     for i in range(0, len(words), chunk_size - overlap):
#         chunk = ' '.join(words[i:i + chunk_size])
#         chunks.append(chunk)
#         if i + chunk_size >= len(words):
#             break
#     return chunks

# def create_embeddings(chunks, embedding_model):
#     return embedding_model.encode(chunks)

# def retrieve_relevant_chunks(query, chunks, embeddings, embedding_model, top_k=3):
#     query_embedding = embedding_model.encode([query])
#     similarities = cosine_similarity(query_embedding, embeddings)[0]
#     top_indices = np.argsort(similarities)[-top_k:][::-1]
#     return [chunks[i] for i in top_indices]

# def generate_response(query, relevant_chunks, gemini_model):
#     context = "\n".join(relevant_chunks)
#     prompt = f"""
#     Based on the following website content, answer the user's question:

#     Context:
#     {context}

#     Question: {query}

#     Please provide a clear and concise answer based on the context provided.
#     You are an AI assistant trained to provide helpful responses.
#     """
#     try:
#         response = gemini_model.generate_content(prompt)
#         return response.text
#     except Exception as e:
#         return f"Error generating response: {str(e)}"

# def speech_to_text(recognizer, microphone):
#     try:
#         with microphone as source:
#             st.info("Listening... Please speak now.")
#             recognizer.adjust_for_ambient_noise(source)
#             audio = recognizer.listen(source, timeout=10, phrase_time_limit=15)
#         st.info("Processing speech...")
#         return recognizer.recognize_google(audio)
#     except sr.WaitTimeoutError:
#         return "Listening timeout. Please try again."
#     except sr.UnknownValueError:
#         return "Could not understand audio. Please try again."
#     except sr.RequestError as e:
#         return f"Error with speech recognition service: {e}"

# def create_tts_audio_file(text):
#     temp_filename = None
#     try:
#         with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
#             temp_filename = tmp_file.name
#         tts_engine = pyttsx3.init()
#         tts_engine.setProperty('rate', 150)
#         tts_engine.setProperty('volume', 0.9)
#         tts_engine.save_to_file(text, temp_filename)
#         tts_engine.runAndWait()
#         tts_engine.stop()
#         return temp_filename
#     except Exception as e:
#         print(f"TTS audio file creation error: {e}")
#         if temp_filename and os.path.exists(temp_filename):
#             os.unlink(temp_filename)
#         return None

# def text_to_speech(text):
#     try:
#         with ThreadPoolExecutor(max_workers=1) as executor:
#             future = executor.submit(create_tts_audio_file, text)
#             temp_filename = future.result(timeout=30)
#             if temp_filename and os.path.exists(temp_filename):
#                 with open(temp_filename, 'rb') as audio_file:
#                     st.audio(audio_file.read(), format='audio/wav', autoplay=True)
#                 os.unlink(temp_filename)
#             else:
#                 st.error("Could not generate audio file.")
#     except Exception as e:
#         st.error(f"Text-to-speech error: {str(e)}")

# def main():
#     st.title("Website Voice Chatbot with RAG")
#     st.write("Ask questions using your voice or text, based on any website!")

#     recognizer, microphone, embedding_model = init_models()

#     st.sidebar.header("Configuration")
#     api_key = st.sidebar.text_input("Enter Gemini API Key:", type="password")

#     if not api_key:
#         st.warning("Please enter your Gemini API key.")
#         return

#     try:
#         gemini_model = setup_gemini(api_key)
#     except Exception as e:
#         st.error(f"Error setting up Gemini: {e}")
#         return

#     website_url = st.sidebar.text_input("Enter Website URL:")

#     if not website_url:
#         st.info("Please enter a valid website URL.")
#         return

#     if 'site_processed' not in st.session_state or st.session_state.get('current_url') != website_url:
#         with st.spinner("Extracting and processing website content..."):
#             site_text = extract_text_from_url(website_url)
#             if site_text:
#                 chunks = chunk_text(site_text)
#                 embeddings = create_embeddings(chunks, embedding_model)
#                 st.session_state.website_chunks = chunks
#                 st.session_state.website_embeddings = embeddings
#                 st.session_state.site_processed = True
#                 st.session_state.current_url = website_url
#                 st.success(f"Website content processed! {len(chunks)} chunks created.")
#             else:
#                 return

#     st.header("Chat Interface")
#     col1, col2 = st.columns(2)

#     with col1:
#         st.subheader("Voice Input")
#         if st.button("Start Voice Input"):
#             user_speech = speech_to_text(recognizer, microphone)
#             st.session_state.last_input = user_speech
#             st.write(f"**You said:** {user_speech}")

#     with col2:
#         st.subheader("Text Input")
#         text_input = st.text_input("Or type your question:")
#         if st.button("Submit Text"):
#             st.session_state.last_input = text_input

#     if 'last_input' in st.session_state and st.session_state.last_input:
#         query = st.session_state.last_input
#         if not query.startswith("Error") and not query.startswith("Could not") and not query.startswith("Listening timeout"):
#             with st.spinner("Generating response..."):
#                 relevant_chunks = retrieve_relevant_chunks(query, st.session_state.website_chunks, st.session_state.website_embeddings, embedding_model)
#                 response = generate_response(query, relevant_chunks, gemini_model)
#                 st.subheader("Response")
#                 st.write(response)

#                 st.subheader("Audio Response")
#                 text_to_speech(response)

#                 if 'chat_history' not in st.session_state:
#                     st.session_state.chat_history = []
#                 st.session_state.chat_history.append({
#                     "query": query,
#                     "response": response,
#                     "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
#                 })

#                 with st.expander("View Relevant Website Chunks"):
#                     for i, chunk in enumerate(relevant_chunks):
#                         st.write(f"**Chunk {i+1}:**")
#                         st.write(chunk[:500] + "..." if len(chunk) > 500 else chunk)
#                         st.write("---")

#     if 'chat_history' in st.session_state:
#         st.header("Chat History")
#         for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):
#             with st.expander(f"Chat {len(st.session_state.chat_history) - i} - {chat['timestamp']}"):
#                 st.write(f"**Q:** {chat['query']}")
#                 st.write(f"**A:** {chat['response']}")

#     col1, col2 = st.columns(2)
#     with col1:
#         if st.button("Clear Current Input"):
#             st.session_state.pop('last_input', None)
#             st.rerun()
#     with col2:
#         if st.button("Clear Chat History"):
#             st.session_state.pop('chat_history', None)
#             st.session_state.pop('last_input', None)
#             st.rerun()

# st.sidebar.markdown("---")
# st.sidebar.header("Instructions")
# st.sidebar.markdown("""
# 1. Enter your Gemini API key  
# 2. Provide a website URL (even if it has expired SSL)  
# 3. Use voice or text to ask questions  
# 4. Listen to AI responses!

# **Required Libraries:**
# ```bash
# pip install streamlit speechrecognition pyttsx3 requests beautifulsoup4 google-generativeai sentence-transformers scikit-learn pyaudio
# """)
# if __name__ == "__main__":
#     main()









# import streamlit as st
# import speech_recognition as sr
# import pyttsx3
# import requests
# from bs4 import BeautifulSoup
# import google.generativeai as genai
# from sentence_transformers import SentenceTransformer
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# import tempfile
# import os
# from concurrent.futures import ThreadPoolExecutor
# import time
# import urllib3

# # Disable SSL warnings
# urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# # 🔒 Hardcoded website list
# HARDCODED_URLS = [
#     # HOME
#     "https://3asolutions.co.in/home.aspx",
#     # COMPANY
#     "https://3asolutions.co.in/aboutus.aspx",
#     "https://3asolutions.co.in/why3asolutions.aspx",
#     # "https://3asolutions.co.in/careers.aspx"
#     # CONTACT US
#     # "https://3asolutions.co.in/contactus.aspx",
#     # INDUSTRIES
#     "https://3asolutions.co.in/industries.aspx",
#     # PRODUCTS
#     "https://3asolutions.co.in/erp.aspx",
#     "https://3asolutions.co.in/hrandpayroll.aspx",
#     "https://3asolutions.co.in/cleaningvalidation.aspx",
#     "https://3asolutions.co.in/labapplication.aspx",
#     "https://3asolutions.co.in/dms.aspx",
#     "https://3asolutions.co.in/ocr.aspx",
#     "https://3asolutions.co.in/rms.aspx",
#     "https://3asolutions.co.in/ecomplaint.aspx",
#     "https://3asolutions.co.in/eschedule.aspx",
#     # SERVICES
#     "https://3asolutions.co.in/softwaredevelopment.aspx",
#     "https://3asolutions.co.in/softwaremaintenance.aspx",
# ]

# @st.cache_resource
# def init_models():
#     recognizer = sr.Recognizer()
#     microphone = sr.Microphone()
#     embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
#     return recognizer, microphone, embedding_model

# def setup_gemini(api_key):
#     genai.configure(api_key=api_key)
#     return genai.GenerativeModel('gemini-1.5-flash')

# def extract_text_from_url(url):
#     try:
#         response = requests.get(url, timeout=10, verify=False)  
#         response.raise_for_status()
#         soup = BeautifulSoup(response.content, "html.parser")
#         return soup.get_text(separator="\n")
#     except Exception as e:
#         st.error(f"Error extracting {url}: {e}")
#         return None

# def chunk_text(text, chunk_size=500, overlap=50):
#     chunks = []
#     words = text.split()
#     for i in range(0, len(words), chunk_size - overlap):
#         chunk = ' '.join(words[i:i + chunk_size])
#         chunks.append(chunk)
#         if i + chunk_size >= len(words):
#             break
#     return chunks

# def create_embeddings(chunks, embedding_model):
#     return embedding_model.encode(chunks)

# def retrieve_relevant_chunks(query, chunks, embeddings, embedding_model, top_k=3):
#     query_embedding = embedding_model.encode([query])
#     similarities = cosine_similarity(query_embedding, embeddings)[0]
#     top_indices = np.argsort(similarities)[-top_k:][::-1]
#     return [chunks[i] for i in top_indices]

# def generate_response(query, relevant_chunks, gemini_model):
#     context = "\n".join(relevant_chunks)
#     prompt = f"""
#     Based on the following website content, answer the user's question:

#     Context:
#     {context}

#     Question: {query}

#     Please provide a clear and concise answer based on the context provided.
#     You are an AI assistant trained to provide helpful responses.
#     If asked about how to contact the company, generate the answer and give this website link: https://3asolutions.co.in/contactus.aspx
#     Ifasked about the careers in the company, generate the answer and give this website link: https://3asolutions.co.in/careers.aspx
#     """
#     try:
#         response = gemini_model.generate_content(prompt)
#         return response.text
#     except Exception as e:
#         return f"Error generating response: {str(e)}"

# def speech_to_text(recognizer, microphone):
#     try:
#         with microphone as source:
#             st.info("Listening... Please speak now.")
#             recognizer.adjust_for_ambient_noise(source)
#             audio = recognizer.listen(source, timeout=10, phrase_time_limit=15)
#         st.info("Processing speech...")
#         return recognizer.recognize_google(audio)
#     except sr.WaitTimeoutError:
#         return "Listening timeout. Please try again."
#     except sr.UnknownValueError:
#         return "Could not understand audio. Please try again."
#     except sr.RequestError as e:
#         return f"Error with speech recognition service: {e}"

# def create_tts_audio_file(text):
#     temp_filename = None
#     try:
#         with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
#             temp_filename = tmp_file.name
#         tts_engine = pyttsx3.init()
#         tts_engine.setProperty('rate', 150)
#         tts_engine.setProperty('volume', 0.9)
#         tts_engine.save_to_file(text, temp_filename)
#         tts_engine.runAndWait()
#         tts_engine.stop()
#         return temp_filename
#     except Exception as e:
#         print(f"TTS audio file creation error: {e}")
#         if temp_filename and os.path.exists(temp_filename):
#             os.unlink(temp_filename)
#         return None

# def text_to_speech(text):
#     try:
#         with ThreadPoolExecutor(max_workers=1) as executor:
#             future = executor.submit(create_tts_audio_file, text)
#             temp_filename = future.result(timeout=30)
#             if temp_filename and os.path.exists(temp_filename):
#                 with open(temp_filename, 'rb') as audio_file:
#                     st.audio(audio_file.read(), format='audio/wav', autoplay=True)
#                 os.unlink(temp_filename)
#             else:
#                 st.error("Could not generate audio file.")
#     except Exception as e:
#         st.error(f"Text-to-speech error: {str(e)}")

# def main():
#     st.title("Multi-Website Voice Chatbot with RAG")
#     st.write("Ask questions using your voice or text, based on multiple websites!")

#     recognizer, microphone, embedding_model = init_models()

#     st.sidebar.header("Configuration")
#     api_key = st.sidebar.text_input("Enter Gemini API Key:", type="password")

#     if not api_key:
#         st.warning("Please enter your Gemini API key.")
#         return

#     try:
#         gemini_model = setup_gemini(api_key)
#     except Exception as e:
#         st.error(f"Error setting up Gemini: {e}")
#         return

#     if 'site_processed' not in st.session_state:
#         with st.spinner("Extracting and processing all websites..."):
#             all_chunks = []
#             for url in HARDCODED_URLS:
#                 site_text = extract_text_from_url(url)
#                 if site_text:
#                     chunks = chunk_text(site_text)
#                     all_chunks.extend(chunks)
#             if all_chunks:
#                 all_embeddings = create_embeddings(all_chunks, embedding_model)
#                 st.session_state.website_chunks = all_chunks
#                 st.session_state.website_embeddings = all_embeddings
#                 st.session_state.site_processed = True
#                 st.success(f"Total {len(all_chunks)} chunks created from {len(HARDCODED_URLS)} websites.")
#             else:
#                 st.error("Failed to process any websites.")
#                 return

#     st.header("Chat Interface")
#     col1, col2 = st.columns(2)

#     with col1:
#         st.subheader("Voice Input")
#         if st.button("Start Voice Input"):
#             user_speech = speech_to_text(recognizer, microphone)
#             st.session_state.last_input = user_speech
#             st.write(f"**You said:** {user_speech}")

#     with col2:
#         st.subheader("Text Input")
#         text_input = st.text_input("Or type your question:")
#         if st.button("Submit Text"):
#             st.session_state.last_input = text_input

#     if 'last_input' in st.session_state and st.session_state.last_input:
#         query = st.session_state.last_input
#         if not query.startswith("Error") and not query.startswith("Could not") and not query.startswith("Listening timeout"):
#             with st.spinner("Generating response..."):
#                 relevant_chunks = retrieve_relevant_chunks(query, st.session_state.website_chunks, st.session_state.website_embeddings, embedding_model)
#                 response = generate_response(query, relevant_chunks, gemini_model)
#                 st.subheader("Response")
#                 st.write(response)

#                 st.subheader("Audio Response")
#                 text_to_speech(response)

#                 if 'chat_history' not in st.session_state:
#                     st.session_state.chat_history = []
#                 st.session_state.chat_history.append({
#                     "query": query,
#                     "response": response,
#                     "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
#                 })

#                 with st.expander("View Relevant Website Chunks"):
#                     for i, chunk in enumerate(relevant_chunks):
#                         st.write(f"**Chunk {i+1}:**")
#                         st.write(chunk[:500] + "..." if len(chunk) > 500 else chunk)
#                         st.write("---")

#     if 'chat_history' in st.session_state:
#         st.header("Chat History")
#         for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):
#             with st.expander(f"Chat {len(st.session_state.chat_history) - i} - {chat['timestamp']}"):
#                 st.write(f"**Q:** {chat['query']}")
#                 st.write(f"**A:** {chat['response']}")

#     col1, col2 = st.columns(2)
#     with col1:
#         if st.button("Clear Current Input"):
#             st.session_state.pop('last_input', None)
#             st.rerun()
#     with col2:
#         if st.button("Clear Chat History"):
#             st.session_state.pop('chat_history', None)
#             st.session_state.pop('last_input', None)
#             st.rerun()

# st.sidebar.markdown("---")
# st.sidebar.header("Instructions")
# st.sidebar.markdown("""
# 1. Enter your Gemini API key  
# 2. Websites are automatically processed  
# 3. Use voice or text to ask questions  
# 4. Listen to AI responses!

# **Required Libraries:**
# ```bash
# pip install streamlit speechrecognition pyttsx3 requests beautifulsoup4 google-generativeai sentence-transformers scikit-learn pyaudio
# """)

# if __name__ == "__main__":
#     main()










# import os
# import requests
# from bs4 import BeautifulSoup
# import google.generativeai as genai
# from sentence_transformers import SentenceTransformer
# import numpy as np
# import chromadb
# from chromadb.config import Settings
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from typing import List
# import urllib3
# import uuid
# import logging

# # Disable SSL warnings
# urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Configuration
# GEMINI_API_KEY = "AIzaSyAVKaYNOZUfMWmIiG8iS2GvjLuW-fS4hEs"  # Replace with your actual API key
# CHROMA_DB_PATH = "./chroma_db"

# # 🔒 Hardcoded website list
# HARDCODED_URLS = [
#     # HOME
#     "https://3asolutions.co.in/home.aspx",
#     # COMPANY
#     "https://3asolutions.co.in/aboutus.aspx",
#     "https://3asolutions.co.in/why3asolutions.aspx",
#     # INDUSTRIES
#     "https://3asolutions.co.in/industries.aspx",
#     # PRODUCTS
#     "https://3asolutions.co.in/erp.aspx",
#     "https://3asolutions.co.in/hrandpayroll.aspx",
#     "https://3asolutions.co.in/cleaningvalidation.aspx",
#     "https://3asolutions.co.in/labapplication.aspx",
#     "https://3asolutions.co.in/dms.aspx",
#     "https://3asolutions.co.in/ocr.aspx",
#     "https://3asolutions.co.in/rms.aspx",
#     "https://3asolutions.co.in/ecomplaint.aspx",
#     "https://3asolutions.co.in/eschedule.aspx",
#     # SERVICES
#     "https://3asolutions.co.in/softwaredevelopment.aspx",
#     "https://3asolutions.co.in/softwaremaintenance.aspx",
# ]

# # Initialize FastAPI app
# app = FastAPI(
#     title="Company Chatbot API",
#     description="Multi-Website RAG Chatbot API using ChromaDB and Gemini",
#     version="1.0.0"
# )

# # Pydantic models
# class QuestionRequest(BaseModel):
#     question: str

# class ChatResponse(BaseModel):
#     answer: str
#     relevant_chunks: List[str]

# # Global variables
# embedding_model = None
# chroma_client = None
# collection = None
# gemini_model = None

# def initialize_models():
#     """Initialize all models and databases"""
#     global embedding_model, chroma_client, collection, gemini_model
    
#     try:
#         # Initialize embedding model
#         embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
#         logger.info("Embedding model initialized successfully")
        
#         # Initialize Gemini
#         genai.configure(api_key=GEMINI_API_KEY)
#         gemini_model = genai.GenerativeModel('gemini-1.5-flash')
#         logger.info("Gemini model initialized successfully")
        
#         # Initialize ChromaDB
#         os.makedirs(CHROMA_DB_PATH, exist_ok=True)
#         chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        
#         # Get or create collection
#         collection_name = "company_knowledge_base"
#         try:
#             collection = chroma_client.get_collection(name=collection_name)
#             logger.info(f"Existing ChromaDB collection '{collection_name}' loaded")
#         except:
#             collection = chroma_client.create_collection(name=collection_name)
#             logger.info(f"New ChromaDB collection '{collection_name}' created")
            
#             # Process and store website data
#             process_and_store_websites()
            
#     except Exception as e:
#         logger.error(f"Error initializing models: {e}")
#         raise

# def extract_text_from_url(url: str) -> str:
#     """Extract text content from a URL"""
#     try:
#         response = requests.get(url, timeout=10, verify=False)
#         response.raise_for_status()
#         soup = BeautifulSoup(response.content, "html.parser")
        
#         # Remove script and style elements
#         for script in soup(["script", "style"]):
#             script.decompose()
            
#         text = soup.get_text(separator="\n")
#         # Clean up the text
#         lines = (line.strip() for line in text.splitlines())
#         chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
#         text = '\n'.join(chunk for chunk in chunks if chunk)
        
#         return text
#     except Exception as e:
#         logger.error(f"Error extracting text from {url}: {e}")
#         return None

# def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
#     """Split text into overlapping chunks"""
#     chunks = []
#     words = text.split()
    
#     for i in range(0, len(words), chunk_size - overlap):
#         chunk = ' '.join(words[i:i + chunk_size])
#         if len(chunk.strip()) > 50:  # Only add meaningful chunks
#             chunks.append(chunk)
#         if i + chunk_size >= len(words):
#             break
            
#     return chunks

# def process_and_store_websites():
#     """Process all websites and store in ChromaDB"""
#     global collection, embedding_model
    
#     logger.info("Starting website processing...")
#     all_chunks = []
#     all_metadatas = []
#     all_ids = []
    
#     for url in HARDCODED_URLS:
#         logger.info(f"Processing: {url}")
#         site_text = extract_text_from_url(url)
        
#         if site_text:
#             chunks = chunk_text(site_text)
#             for i, chunk in enumerate(chunks):
#                 chunk_id = str(uuid.uuid4())
#                 all_chunks.append(chunk)
#                 all_metadatas.append({
#                     "source_url": url,
#                     "chunk_index": i,
#                     "chunk_id": chunk_id
#                 })
#                 all_ids.append(chunk_id)
    
#     if all_chunks:
#         # Generate embeddings
#         logger.info(f"Generating embeddings for {len(all_chunks)} chunks...")
#         embeddings = embedding_model.encode(all_chunks).tolist()
        
#         # Store in ChromaDB
#         collection.add(
#             embeddings=embeddings,
#             documents=all_chunks,
#             metadatas=all_metadatas,
#             ids=all_ids
#         )
        
#         logger.info(f"Successfully stored {len(all_chunks)} chunks in ChromaDB")
#     else:
#         logger.error("No chunks were created from the websites")

# def retrieve_relevant_chunks(query: str, top_k: int = 3) -> List[str]:
#     """Retrieve relevant chunks from ChromaDB"""
#     global collection
    
#     try:
#         results = collection.query(
#             query_texts=[query],
#             n_results=top_k
#         )
        
#         return results['documents'][0] if results['documents'] else []
#     except Exception as e:
#         logger.error(f"Error retrieving chunks: {e}")
#         return []

# def generate_response(query: str, relevant_chunks: List[str]) -> str:
#     """Generate response using Gemini"""
#     global gemini_model
    
#     context = "\n".join(relevant_chunks)
#     prompt = f"""
#     Based on the following website content, answer the user's question:

#     Context:
#     {context}

#     Question: {query}

#     Please provide a clear and concise answer based on the context provided.
#     You are an AI assistant trained to provide helpful responses about 3A Solutions company.
#     If asked about how to contact the company, generate the answer and give this website link: https://3asolutions.co.in/contactus.aspx
#     If asked about careers in the company, generate the answer and give this website link: https://3asolutions.co.in/careers.aspx
    
#     If the context doesn't contain relevant information to answer the question, politely say that you don't have enough information about that specific topic in the available content.
#     """
    
#     try:
#         response = gemini_model.generate_content(prompt)
#         return response.text
#     except Exception as e:
#         logger.error(f"Error generating response: {e}")
#         return "I apologize, but I'm having trouble generating a response right now. Please try again later."

# @app.on_event("startup")
# async def startup_event():
#     """Initialize models and database on startup"""
#     logger.info("Starting up the application...")
#     initialize_models()
#     logger.info("Application startup completed successfully")

# @app.get("/")
# async def root():
#     """Root endpoint"""
#     return {"message": "Company Chatbot API is running", "version": "1.0.0"}

# @app.get("/health")
# async def health_check():
#     """Health check endpoint"""
#     return {"status": "healthy", "chroma_db_path": CHROMA_DB_PATH}

# @app.post("/ask", response_model=ChatResponse)
# async def ask_question(request: QuestionRequest):
#     """Main endpoint to ask questions and get answers"""
#     try:
#         if not request.question.strip():
#             raise HTTPException(status_code=400, detail="Question cannot be empty")
        
#         logger.info(f"Received question: {request.question}")
        
#         # Retrieve relevant chunks
#         relevant_chunks = retrieve_relevant_chunks(request.question, top_k=3)
        
#         if not relevant_chunks:
#             return ChatResponse(
#                 answer="I apologize, but I couldn't find relevant information to answer your question. Please try rephrasing your question or ask about our company's products, services, or general information.",
#                 relevant_chunks=[]
#             )
        
#         # Generate response
#         answer = generate_response(request.question, relevant_chunks)
        
#         logger.info("Response generated successfully")
        
#         return ChatResponse(
#             answer=answer,
#             relevant_chunks=relevant_chunks[:3]  # Return top 3 chunks for reference
#         )
        
#     except Exception as e:
#         logger.error(f"Error processing question: {e}")
#         raise HTTPException(status_code=500, detail="Internal server error occurred while processing your question")

# @app.get("/collection-info")
# async def get_collection_info():
#     """Get information about the ChromaDB collection"""
#     try:
#         count = collection.count()
#         return {
#             "collection_name": collection.name,
#             "total_chunks": count,
#             "chroma_db_path": CHROMA_DB_PATH,
#             "processed_urls": len(HARDCODED_URLS)
#         }
#     except Exception as e:
#         logger.error(f"Error getting collection info: {e}")
#         raise HTTPException(status_code=500, detail="Error retrieving collection information")

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000)






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
