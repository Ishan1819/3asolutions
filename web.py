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









import streamlit as st
import speech_recognition as sr
import pyttsx3
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import tempfile
import os
from concurrent.futures import ThreadPoolExecutor
import time
import urllib3

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 🔒 Hardcoded website list
HARDCODED_URLS = [
    # HOME
    "https://3asolutions.co.in/home.aspx",
    # COMPANY
    "https://3asolutions.co.in/aboutus.aspx",
    "https://3asolutions.co.in/why3asolutions.aspx",
    # "https://3asolutions.co.in/careers.aspx"
    # CONTACT US
    # "https://3asolutions.co.in/contactus.aspx",
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

@st.cache_resource
def init_models():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return recognizer, microphone, embedding_model

def setup_gemini(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-1.5-flash')

def extract_text_from_url(url):
    try:
        response = requests.get(url, timeout=10, verify=False)  
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        return soup.get_text(separator="\n")
    except Exception as e:
        st.error(f"Error extracting {url}: {e}")
        return None

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    words = text.split()
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks

def create_embeddings(chunks, embedding_model):
    return embedding_model.encode(chunks)

def retrieve_relevant_chunks(query, chunks, embeddings, embedding_model, top_k=3):
    query_embedding = embedding_model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

def generate_response(query, relevant_chunks, gemini_model):
    context = "\n".join(relevant_chunks)
    prompt = f"""
    Based on the following website content, answer the user's question:

    Context:
    {context}

    Question: {query}

    Please provide a clear and concise answer based on the context provided.
    You are an AI assistant trained to provide helpful responses.
    If asked about how to contact the company, generate the answer and give this website link: https://3asolutions.co.in/contactus.aspx
    Ifasked about the careers in the company, generate the answer and give this website link: https://3asolutions.co.in/careers.aspx
    """
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"

def speech_to_text(recognizer, microphone):
    try:
        with microphone as source:
            st.info("Listening... Please speak now.")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=15)
        st.info("Processing speech...")
        return recognizer.recognize_google(audio)
    except sr.WaitTimeoutError:
        return "Listening timeout. Please try again."
    except sr.UnknownValueError:
        return "Could not understand audio. Please try again."
    except sr.RequestError as e:
        return f"Error with speech recognition service: {e}"

def create_tts_audio_file(text):
    temp_filename = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            temp_filename = tmp_file.name
        tts_engine = pyttsx3.init()
        tts_engine.setProperty('rate', 150)
        tts_engine.setProperty('volume', 0.9)
        tts_engine.save_to_file(text, temp_filename)
        tts_engine.runAndWait()
        tts_engine.stop()
        return temp_filename
    except Exception as e:
        print(f"TTS audio file creation error: {e}")
        if temp_filename and os.path.exists(temp_filename):
            os.unlink(temp_filename)
        return None

def text_to_speech(text):
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(create_tts_audio_file, text)
            temp_filename = future.result(timeout=30)
            if temp_filename and os.path.exists(temp_filename):
                with open(temp_filename, 'rb') as audio_file:
                    st.audio(audio_file.read(), format='audio/wav', autoplay=True)
                os.unlink(temp_filename)
            else:
                st.error("Could not generate audio file.")
    except Exception as e:
        st.error(f"Text-to-speech error: {str(e)}")

def main():
    st.title("Multi-Website Voice Chatbot with RAG")
    st.write("Ask questions using your voice or text, based on multiple websites!")

    recognizer, microphone, embedding_model = init_models()

    st.sidebar.header("Configuration")
    api_key = st.sidebar.text_input("Enter Gemini API Key:", type="password")

    if not api_key:
        st.warning("Please enter your Gemini API key.")
        return

    try:
        gemini_model = setup_gemini(api_key)
    except Exception as e:
        st.error(f"Error setting up Gemini: {e}")
        return

    if 'site_processed' not in st.session_state:
        with st.spinner("Extracting and processing all websites..."):
            all_chunks = []
            for url in HARDCODED_URLS:
                site_text = extract_text_from_url(url)
                if site_text:
                    chunks = chunk_text(site_text)
                    all_chunks.extend(chunks)
            if all_chunks:
                all_embeddings = create_embeddings(all_chunks, embedding_model)
                st.session_state.website_chunks = all_chunks
                st.session_state.website_embeddings = all_embeddings
                st.session_state.site_processed = True
                st.success(f"Total {len(all_chunks)} chunks created from {len(HARDCODED_URLS)} websites.")
            else:
                st.error("Failed to process any websites.")
                return

    st.header("Chat Interface")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Voice Input")
        if st.button("Start Voice Input"):
            user_speech = speech_to_text(recognizer, microphone)
            st.session_state.last_input = user_speech
            st.write(f"**You said:** {user_speech}")

    with col2:
        st.subheader("Text Input")
        text_input = st.text_input("Or type your question:")
        if st.button("Submit Text"):
            st.session_state.last_input = text_input

    if 'last_input' in st.session_state and st.session_state.last_input:
        query = st.session_state.last_input
        if not query.startswith("Error") and not query.startswith("Could not") and not query.startswith("Listening timeout"):
            with st.spinner("Generating response..."):
                relevant_chunks = retrieve_relevant_chunks(query, st.session_state.website_chunks, st.session_state.website_embeddings, embedding_model)
                response = generate_response(query, relevant_chunks, gemini_model)
                st.subheader("Response")
                st.write(response)

                st.subheader("Audio Response")
                text_to_speech(response)

                if 'chat_history' not in st.session_state:
                    st.session_state.chat_history = []
                st.session_state.chat_history.append({
                    "query": query,
                    "response": response,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                })

                with st.expander("View Relevant Website Chunks"):
                    for i, chunk in enumerate(relevant_chunks):
                        st.write(f"**Chunk {i+1}:**")
                        st.write(chunk[:500] + "..." if len(chunk) > 500 else chunk)
                        st.write("---")

    if 'chat_history' in st.session_state:
        st.header("Chat History")
        for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):
            with st.expander(f"Chat {len(st.session_state.chat_history) - i} - {chat['timestamp']}"):
                st.write(f"**Q:** {chat['query']}")
                st.write(f"**A:** {chat['response']}")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Current Input"):
            st.session_state.pop('last_input', None)
            st.rerun()
    with col2:
        if st.button("Clear Chat History"):
            st.session_state.pop('chat_history', None)
            st.session_state.pop('last_input', None)
            st.rerun()

st.sidebar.markdown("---")
st.sidebar.header("Instructions")
st.sidebar.markdown("""
1. Enter your Gemini API key  
2. Websites are automatically processed  
3. Use voice or text to ask questions  
4. Listen to AI responses!

**Required Libraries:**
```bash
pip install streamlit speechrecognition pyttsx3 requests beautifulsoup4 google-generativeai sentence-transformers scikit-learn pyaudio
""")

if __name__ == "__main__":
    main()
