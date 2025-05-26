
import streamlit as st
import speech_recognition as sr
import pyttsx3
import requests
import PyPDF2
from io import BytesIO
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import threading
import time
import tempfile
import os
from concurrent.futures import ThreadPoolExecutor
import queue

# Initialize global variables
@st.cache_resource
def init_models():
    """Initialize speech recognition, TTS, and embedding models"""
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    return recognizer, microphone, embedding_model

def setup_gemini(api_key):
    """Setup Gemini API"""
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-1.5-flash')

def extract_pdf_from_url(pdf_url):
    """Extract text from PDF URL"""
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()
        
        pdf_file = BytesIO(response.content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + "\n"
        
        return text
    except Exception as e:
        st.error(f"Error extracting PDF: {str(e)}")
        return None

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into chunks for RAG"""
    chunks = []
    words = text.split()
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
        
        if i + chunk_size >= len(words):
            break
    
    return chunks

def create_embeddings(chunks, embedding_model):
    """Create embeddings for text chunks"""
    embeddings = embedding_model.encode(chunks)
    return embeddings

def retrieve_relevant_chunks(query, chunks, embeddings, embedding_model, top_k=3):
    """Retrieve most relevant chunks based on query"""
    query_embedding = embedding_model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    
    # Get top-k most similar chunks
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    relevant_chunks = [chunks[i] for i in top_indices]
    
    return relevant_chunks

def generate_response(query, relevant_chunks, gemini_model):
    """Generate response using Gemini with RAG context"""
    context = "\n".join(relevant_chunks)
    
    prompt = f"""
    Based on the following context from the PDF document, answer the user's question:
    
    Context:
    {context}
    
    Question: {query}
    
    Please provide a clear and concise answer based on the context provided. If the answer is not found in the context, please say so.
    You are an AI assistant trained to provide accurate and helpful responses made by Ishan Patil.
    If someone asks you about which person made you, please say "I was made by Ishan Patil, a student at PCCOE, Pune."
    """
    
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"

def speech_to_text(recognizer, microphone):
    """Convert speech to text"""
    try:
        with microphone as source:
            st.info("Listening... Please speak now.")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=15)
        
        st.info("Processing speech...")
        text = recognizer.recognize_google(audio)
        return text
    except sr.WaitTimeoutError:
        return "Listening timeout. Please try again."
    except sr.UnknownValueError:
        return "Could not understand audio. Please try again."
    except sr.RequestError as e:
        return f"Error with speech recognition service: {e}"

def create_tts_audio_file(text):
    """Create TTS audio file in a separate thread to avoid event loop conflicts"""
    temp_filename = None
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            temp_filename = tmp_file.name
        
        # Create TTS engine in isolated context
        tts_engine = pyttsx3.init()
        tts_engine.setProperty('rate', 150)
        tts_engine.setProperty('volume', 0.9)
        
        # Save to file
        tts_engine.save_to_file(text, temp_filename)
        tts_engine.runAndWait()
        
        # Properly cleanup the engine
        try:
            tts_engine.stop()
            del tts_engine
        except:
            pass
        
        return temp_filename
        
    except Exception as e:
        print(f"TTS audio file creation error: {e}")
        if temp_filename and os.path.exists(temp_filename):
            try:
                os.unlink(temp_filename)
            except:
                pass
        return None

def text_to_speech(text):
    """Convert text to speech with improved error handling"""
    try:
        # Use ThreadPoolExecutor to avoid event loop conflicts
        with ThreadPoolExecutor(max_workers=1) as executor:
            # Submit TTS task to thread pool
            future = executor.submit(create_tts_audio_file, text)
            
            # Get result with timeout
            temp_filename = future.result(timeout=30)
            
            if temp_filename and os.path.exists(temp_filename):
                try:
                    # Display audio player in Streamlit
                    with open(temp_filename, 'rb') as audio_file:
                        st.audio(audio_file.read(), format='audio/wav', autoplay=True)
                    
                    # Clean up temp file
                    os.unlink(temp_filename)
                    
                except Exception as audio_error:
                    st.error(f"Error playing audio: {audio_error}")
                    # Clean up on error
                    try:
                        if temp_filename:
                            os.unlink(temp_filename)
                    except:
                        pass
            else:
                st.error("Could not generate audio file")
                
    except Exception as e:
        st.error(f"Text-to-speech error: {str(e)}")
        
        # Simple fallback without threading
        try:
            st.info("Trying fallback TTS method...")
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                fallback_filename = tmp_file.name
            
            fallback_tts = pyttsx3.init()
            fallback_tts.setProperty('rate', 150)
            fallback_tts.setProperty('volume', 0.9)
            fallback_tts.save_to_file(text, fallback_filename)
            fallback_tts.runAndWait()
            
            # Cleanup engine immediately
            try:
                fallback_tts.stop()
                del fallback_tts
            except:
                pass
            
            if os.path.exists(fallback_filename):
                with open(fallback_filename, 'rb') as audio_file:
                    st.audio(audio_file.read(), format='audio/wav')
                os.unlink(fallback_filename)
                st.info("Audio ready - click play button above to listen")
            
        except Exception as fallback_error:
            st.error(f"Fallback TTS also failed: {fallback_error}")
            st.info("Text-to-speech not available. Response shown as text only.")

def main():
    st.title("🎤 Speech-to-Speech RAG PDF Chatbot")
    st.write("Upload a PDF via URL and chat with it using voice!")
    
    # Initialize models
    recognizer, microphone, embedding_model = init_models()
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Gemini API Key
    api_key = st.sidebar.text_input("Enter Gemini API Key:", type="password")
    
    if not api_key:
        st.warning("Please enter your Gemini API key to continue.")
        st.info("Get your API key from: https://makersuite.google.com/app/apikey")
        return
    
    # Setup Gemini
    try:
        gemini_model = setup_gemini(api_key)
    except Exception as e:
        st.error(f"Error setting up Gemini: {e}")
        return
    
    # PDF URL input
    pdf_url = st.sidebar.text_input("Enter PDF URL:")
    
    if not pdf_url:
        st.info("Please enter a PDF URL to start.")
        return
    
    # Process PDF
    if 'pdf_processed' not in st.session_state or st.session_state.get('current_pdf_url') != pdf_url:
        with st.spinner("Extracting and processing PDF..."):
            pdf_text = extract_pdf_from_url(pdf_url)
            
            if pdf_text:
                chunks = chunk_text(pdf_text)
                embeddings = create_embeddings(chunks, embedding_model)
                
                # Store in session state
                st.session_state.pdf_chunks = chunks
                st.session_state.pdf_embeddings = embeddings
                st.session_state.pdf_processed = True
                st.session_state.current_pdf_url = pdf_url
                
                st.success(f"PDF processed successfully! Created {len(chunks)} chunks.")
            else:
                st.error("Failed to process PDF. Please check the URL.")
                return
    
    # Chat interface
    st.header("Chat Interface")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎤 Voice Input")
        if st.button("Start Voice Input", type="primary"):
            user_speech = speech_to_text(recognizer, microphone)
            st.session_state.last_speech_input = user_speech
            st.write(f"**You said:** {user_speech}")
    
    with col2:
        st.subheader("⌨️ Text Input")
        text_input = st.text_input("Or type your question:")
        if st.button("Submit Text"):
            st.session_state.last_speech_input = text_input
    
    # Process query
    if 'last_speech_input' in st.session_state and st.session_state.last_speech_input:
        query = st.session_state.last_speech_input
        
        if not query.startswith("Error") and not query.startswith("Could not") and not query.startswith("Listening timeout"):
            with st.spinner("Generating response..."):
                try:
                    # Retrieve relevant chunks
                    relevant_chunks = retrieve_relevant_chunks(
                        query, 
                        st.session_state.pdf_chunks, 
                        st.session_state.pdf_embeddings, 
                        embedding_model
                    )
                    
                    # Generate response
                    response = generate_response(query, relevant_chunks, gemini_model)
                    
                    # Display text response
                    st.subheader("📝 Response")
                    st.write(response)
                    
                    # Add to chat history
                    if 'chat_history' not in st.session_state:
                        st.session_state.chat_history = []
                    
                    st.session_state.chat_history.append({
                        "query": query,
                        "response": response,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    })
                    
                    # Convert response to speech
                    st.subheader("🔊 Audio Response")
                    text_to_speech(response)
                    
                    # Show relevant chunks (optional)
                    with st.expander("View Relevant PDF Chunks"):
                        for i, chunk in enumerate(relevant_chunks):
                            st.write(f"**Chunk {i+1}:**")
                            st.write(chunk[:500] + "..." if len(chunk) > 500 else chunk)
                            st.write("---")
                            
                except Exception as e:
                    st.error(f"Error processing query: {e}")
    
    # Chat history display
    if 'chat_history' in st.session_state and st.session_state.chat_history:
        st.header("💬 Chat History")
        for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):  # Show last 5 chats
            with st.expander(f"Chat {len(st.session_state.chat_history) - i} - {chat['timestamp']}"):
                st.write(f"**Q:** {chat['query']}")
                st.write(f"**A:** {chat['response']}")
    
    # Control buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Current Input"):
            if 'last_speech_input' in st.session_state:
                del st.session_state.last_speech_input
            st.rerun()
    
    with col2:
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            if 'last_speech_input' in st.session_state:
                del st.session_state.last_speech_input
            st.rerun()

# Instructions
st.sidebar.markdown("---")
st.sidebar.header("Instructions")
st.sidebar.markdown("""
1. Enter your Gemini API key
2. Provide a PDF URL
3. Use voice input or text input to ask questions
4. Listen to the AI response
5. Continue the conversation!

**Required Libraries:**
```bash
pip install streamlit speechrecognition pyttsx3 
pip install requests PyPDF2 google-generativeai 
pip install sentence-transformers scikit-learn
pip install pyaudio  # For microphone access
```

**Troubleshooting:**
- If TTS doesn't work, try refreshing the page
- Ensure your microphone permissions are granted
- Check your internet connection for API calls
""")

if __name__ == "__main__":
    main()