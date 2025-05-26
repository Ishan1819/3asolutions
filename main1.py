# import os
# import fitz  # PyMuPDF
# import requests
# import chromadb
# from chromadb.utils import embedding_functions
# from sentence_transformers import SentenceTransformer
# import autogen
# from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
# import google.generativeai as genai
# from dotenv import load_dotenv

# # Load environment variables from .env
# load_dotenv()
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # === Step 1: Load PDF from URL ===
# def download_pdf_text(pdf_url):
#     response = requests.get(pdf_url)
#     with open("weather_doc.pdf", "wb") as f:
#         f.write(response.content)

#     doc = fitz.open("weather_doc.pdf")
#     text = ""
#     for page in doc:
#         text += page.get_text()
#     return text

# # === Step 2: Embed PDF to ChromaDB ===
# def create_vector_store(text_data):
#     client = chromadb.Client()
#     embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
#     collection = client.create_collection(name="weather_docs", embedding_function=embedding_fn)
#     print("Creating vector store with ChromaDB...")
#     chunks = [text_data[i:i+500] for i in range(0, len(text_data), 500)]
#     for i, chunk in enumerate(chunks):
#         collection.add(documents=[chunk], ids=[f"doc_{i}"])
#     return collection

# # === Step 3: Define Retrieval Tool ===
# def retrieve_from_pdf(query, collection):
#     results = collection.query(query_texts=[query], n_results=3)
#     return "\n".join([doc for doc in results['documents'][0]])

# # === Step 4: Weather API Tool ===
# def get_weather(city_name):
#     api_key = os.getenv("WEATHER_API_KEY")
#     url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={city_name}&aqi=no"
#     response = requests.get(url)
#     if response.status_code == 200:
#         data = response.json()
#         return (
#             f"Weather in {city_name}:\n"
#             f"Temperature: {data['current']['temp_c']}°C\n"
#             f"Condition: {data['current']['condition']['text']}"
#         )
#     else:
#         return "Could not retrieve weather data."

# # === Step 5: Define Agents ===

# # Agent 1 - Retrieval
# retrieval_agent = AssistantAgent(
#     name="retriever",
#     llm_config={"model": "models/gemini-1.5-pro", "api_key": os.getenv("GOOGLE_API_KEY")},
#     code_execution_config=False
# )

# # Agent 2 - Weather Forecaster
# forecast_agent = AssistantAgent(
#     name="forecaster",
#     function_map={"get_weather": get_weather},
#     llm_config=False,
#     code_execution_config={"use_docker": False}
# )

# # === Step 6: User Proxy with Tool Routing ===

# class SmartUserProxy(UserProxyAgent):
#     def __init__(self, collection, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.collection = collection

#     def _is_weather_query(self, message):
#         keywords = ["weather", "temperature", "climate", "forecast"]
#         return any(k in message.lower() for k in keywords) and "in" in message.lower()

#     def _extract_city(self, message):
#         parts = message.lower().split("in")
#         return parts[-1].strip().capitalize() if len(parts) > 1 else "Mumbai"

#     def react(self, message, sender, config):
#         if self._is_weather_query(message):
#             city = self._extract_city(message)
#             print(f"⛅ Detected weather query for city: {city}")
#             result = get_weather(city)
#             print(result)
#         else:
#             print(f"📚 General query: retrieving RAG data...")
#             context = retrieve_from_pdf(message, self.collection)

#             model = genai.GenerativeModel("models/gemini-1.5-pro")
#             response = model.generate_content(f"Based on the following document context:\n{context}\n\nAnswer the user's question: {message}")
#             print(response.text)

# # === Step 7: Group Chat ===
# def start_group_chat(collection):
#     user = SmartUserProxy(
#         name="user_proxy",
#         collection=collection,
#         human_input_mode="ALWAYS",
#         code_execution_config={"use_docker": False},
#         is_termination_msg=lambda x: x.lower() == "exit"
#     )
#     print("Starting group chat with agents...")

#     group_chat = GroupChat(agents=[user, retrieval_agent, forecast_agent], messages=[], max_round=10)
#     manager = GroupChatManager(groupchat=group_chat, llm_config={"model": "models/gemini-1.5-pro", "api_key": os.getenv("GOOGLE_API_KEY")})

#     user.initiate_chat(manager)

# # === Run it ===
# if __name__ == "__main__":
#     pdf_url = input("Enter PDF URL to ingest weather-related knowledge: ")
#     pdf_text = download_pdf_text(pdf_url)
#     collection = create_vector_store(pdf_text)
#     start_group_chat(collection)










# import os
# import requests
# from io import BytesIO

# import chromadb
# from sentence_transformers import SentenceTransformer
# from PyPDF2 import PdfReader

# from crewai import Agent, Task, Crew
# from langchain_google_genai import GoogleGenerativeAI

# # ---------- Configuration ----------
# GEMINI_API_KEY = "AIzaSyDL6czoKfPp7ljkStGOiQqSZvdaf_eOVAw"
# WEATHER_API_KEY = "c84fe43a8819a5c07274071823523e18"
# PDF_URL = "https://internal.imd.gov.in/section/nhac/dynamic/allindianew.pdf"


# from crewai import Tool

# class WeatherTool(Tool):
#     def __init__(self, api_key):
#         super().__init__(name="WeatherTool")
#         self.api_key = api_key

#     def run(self, location: str) -> str:
#         url = f"http://api.weatherapi.com/v1/current.json?key={self.api_key}&q={location}"
#         response = requests.get(url)
#         if response.status_code == 200:
#             data = response.json()
#             return f"Current weather in {location}:\n" \
#                    f"Temperature: {data['current']['temp_c']}°C\n" \
#                    f"Condition: {data['current']['condition']['text']}\n" \
#                    f"Humidity: {data['current']['humidity']}%\n" \
#                    f"Wind: {data['current']['wind_kph']} kph"
#         else:
#             return "Weather data not available."


# class RagTool(Tool):
#     def __init__(self, collection, embedder):
#         super().__init__(name="RAGTool")
#         self.collection = collection
#         self.embedder = embedder

#     def run(self, query: str) -> str:
#         embedding = self.embedder.encode([query])[0]
#         results = self.collection.query(query_embeddings=[embedding], n_results=1)
#         return results['documents'][0][0] if results['documents'] else "No relevant information found."

# # ---------- Step 1: Load and Extract Text from PDF ----------
# def download_and_extract_text(pdf_url):
#     response = requests.get(pdf_url)
#     response.raise_for_status()
#     file_stream = BytesIO(response.content)
#     reader = PdfReader(file_stream)
#     text = " ".join([page.extract_text() or "" for page in reader.pages])
#     return text

# # ---------- Step 2: Setup ChromaDB Vector Store ----------
# def setup_chroma_from_text(text):
#     client = chromadb.Client()
#     embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

#     # Split text into chunks (1000 chars)
#     chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
#     embeddings = embedding_model.encode(chunks).tolist()

#     # Create collection or get if exists
#     try:
#         collection = client.get_collection(name="weather_rag")
#         # Clear existing data
#         collection.delete(where={})
#     except Exception:
#         collection = client.create_collection(name="weather_rag")

#     collection.add(documents=chunks, embeddings=embeddings, ids=[str(i) for i in range(len(chunks))])
#     return collection, embedding_model

# # ---------- Step 3: Define RAG retrieval tool ----------
# def rag_tool_instance(query):
#     embedding = embedder.encode([query]).tolist()
#     results = collection.query(query_embeddings=embedding, n_results=3)
#     if results['documents'] and len(results['documents'][0]) > 0:
#         return "\n".join(results['documents'][0])
#     return "No relevant information found."

# # ---------- Step 4: Define Weather API Tool ----------
# def weather_tool(location):
#     url = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={location}"
#     try:
#         response = requests.get(url)
#         response.raise_for_status()
#         data = response.json()
#         return (
#             f"Current weather in {location}:\n"
#             f"Temperature: {data['current']['temp_c']}°C\n"
#             f"Condition: {data['current']['condition']['text']}\n"
#             f"Humidity: {data['current']['humidity']}%\n"
#             f"Wind: {data['current']['wind_kph']} kph"
#         )
#     except Exception:
#         return "Weather data not available."
# pdf_text = download_and_extract_text(PDF_URL)
# collection, embedder = setup_chroma_from_text(pdf_text)
# # ---------- Step 5: Setup Agents properly (tools as callable list) ----------
# # Wrap tools as callable
# rag_tool = RagTool(collection, embedder)

# retrieval_agent = Agent(
#     role="retrieval expert",
#     goal="answer user questions based on PDF content using RAG",
#     tools=[rag_tool],
#     backstory="You specialize in answering questions by retrieving relevant information from embedded documents."
# )


# forecasting_agent = Agent(
#     role="weather forecaster",
#     goal="provide real-time weather updates for any location in India",
#     tools=[weather_tool],
#     backstory="You provide real-time weather information using the weather API.",
# )

# # ---------- Initialize embeddings and collection globally ----------



# # ---------- Step 6: Main interaction logic ----------
# def ask_question(user_input):
#     # Determine agent based on user input
#     if "weather in" in user_input.lower():
#         location = user_input.lower().split("weather in")[-1].strip()
#         task = Task(description=f"Get current weather in {location}", agent=forecasting_agent)
#     else:
#         task = Task(description=f"Answer the question: {user_input}", agent=retrieval_agent)

#     crew = Crew(agents=[retrieval_agent, forecasting_agent], tasks=[task])
#     raw_result = crew.kickoff()

#     # Use Gemini LLM to polish the output
#     gemini_llm = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)
#     polished_response = gemini_llm.invoke(f"Refine this response for the user:\n{raw_result}")

#     return polished_response.text if hasattr(polished_response, 'text') else polished_response

# # ---------- Step 7: Run interactive chat ----------
# if __name__ == "__main__":
#     print("Type 'exit' or 'quit' to stop.")
#     while True:
#         query = input("Ask about Indian weather or other queries: ")
#         if query.lower() in ['exit', 'quit']:
#             break
#         answer = ask_question(query)
#         print("\nAnswer:", answer)











# from crewai import Agent, Task, Crew
# from sentence_transformers import SentenceTransformer
# import chromadb
# import requests
# from io import BytesIO
# from PyPDF2 import PdfReader
# from crewai_tools import tools
# from langchain_core.tools import BaseTool
# from typing import Type
# from pydantic import BaseModel, Field

# # --- Your API keys ---
# GEMINI_API_KEY = "AIzaSyDL6czoKfPp7ljkStGOiQqSZvdaf_eOVAw"
# WEATHER_API_KEY = "c84fe43a8819a5c07274071823523e18"
# PDF_URL = "https://internal.imd.gov.in/section/nhac/dynamic/allindianew.pdf"

# # --- Download and extract PDF text ---
# def download_and_extract_text(pdf_url):
#     response = requests.get(pdf_url)
#     file_stream = BytesIO(response.content)
#     reader = PdfReader(file_stream)
#     text = " ".join([page.extract_text() or "" for page in reader.pages])
#     return text

# # --- Setup Chroma and embeddings ---
# pdf_text = download_and_extract_text(PDF_URL)
# chroma_client = chromadb.Client()
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# texts = [pdf_text[i:i+1000] for i in range(0, len(pdf_text), 1000)]
# embeddings = embedding_model.encode(texts)

# try:
#     chroma_client.delete_collection(name="weather_rag")
# except:
#     pass  # Collection doesn't exist, that's fine

# collection = chroma_client.create_collection(name="weather_rag")
# collection.add(documents=texts, embeddings=embeddings, ids=[str(i) for i in range(len(texts))])

# # --- Define Input Schemas ---
# class RagInput(BaseModel):
#     query: str = Field(description="The query to search for in the PDF documents")

# class WeatherInput(BaseModel):
#     location: str = Field(description="The location to get weather information for")

# # --- Global references for tools ---
# global_collection = None
# global_embedder = None
# global_weather_api_key = None

# # --- Define Tools ---
# class RagTool(BaseTool):
#     name: str = "rag_search"
#     description: str = "Search for information in PDF documents using RAG (Retrieval Augmented Generation)"
#     args_schema: Type[BaseModel] = RagInput

#     def _run(self, query: str) -> str:
#         """Search for relevant information in the PDF documents"""
#         try:
#             embedding = global_embedder.encode([query])[0]
#             results = global_collection.query(query_embeddings=[embedding], n_results=3)
            
#             if results['documents'] and results['documents'][0]:
#                 # Combine top results for better context
#                 combined_results = "\n\n".join(results['documents'][0])
#                 return f"Found relevant information:\n{combined_results}"
#             else:
#                 return "No relevant information found in the PDF documents."
#         except Exception as e:
#             return f"Error searching documents: {str(e)}"

# class WeatherTool(BaseTool):
#     name: str = "get_weather"
#     description: str = "Get current weather information for a specific location"
#     args_schema: Type[BaseModel] = WeatherInput

#     def _run(self, location: str) -> str:
#         """Get current weather for the specified location"""
#         try:
#             url = f"http://api.weatherapi.com/v1/current.json?key={global_weather_api_key}&q={location}"
#             response = requests.get(url)
            
#             if response.status_code == 200:
#                 data = response.json()
#                 return f"Current weather in {location}:\n" \
#                        f"Temperature: {data['current']['temp_c']}°C\n" \
#                        f"Condition: {data['current']['condition']['text']}\n" \
#                        f"Humidity: {data['current']['humidity']}%\n" \
#                        f"Wind: {data['current']['wind_kph']} kph"
#             else:
#                 return f"Weather data not available for {location}. Status code: {response.status_code}"
#         except Exception as e:
#             return f"Error fetching weather data: {str(e)}"

# # Set global references
# global_collection = collection
# global_embedder = embedding_model
# global_weather_api_key = WEATHER_API_KEY

# # Initialize tools
# rag_tool = RagTool()
# weather_tool = WeatherTool()

# # --- Define Agents ---
# retrieval_agent = Agent(
#     role="Weather Information Retrieval Expert",
#     goal="Answer user questions about weather patterns and forecasts based on PDF content using RAG",
#     tools=[rag_tool],
#     backstory="You are an expert in weather analysis who specializes in retrieving and analyzing weather information from official meteorological documents and reports.",
#     verbose=True,
#     allow_delegation=False
# )

# forecasting_agent = Agent(
#     role="Real-time Weather Forecast Agent",
#     goal="Provide current weather conditions and forecasts using live weather API data",
#     backstory="You are a weather forecasting specialist who provides real-time weather updates and current conditions for any location using live weather data APIs.",
#     verbose=True,
#     allow_delegation=False,
#     tools=[weather_tool]
# )

# # --- Main function ---
# def ask_question(user_input):
#     """Process user input and route to appropriate agent"""
#     user_input_lower = user_input.lower()
    
#     if any(keyword in user_input_lower for keyword in ["current weather", "weather in", "weather for", "temperature in"]):
#         # Extract location for weather query
#         location = user_input
#         for phrase in ["current weather in", "weather in", "weather for", "temperature in"]:
#             if phrase in user_input_lower:
#                 location = user_input_lower.split(phrase)[-1].strip()
#                 break
        
#         task = Task(
#             description=f"Get current weather information for {location}",
#             agent=forecasting_agent,
#             expected_output="Current weather conditions including temperature, humidity, wind speed, and weather condition"
#         )
#     else:
#         # Use RAG for general weather questions
#         task = Task(
#             description=f"Search for and provide information about: {user_input}",
#             agent=retrieval_agent,
#             expected_output="Relevant information from weather documents and reports"
#         )

#     crew = Crew(
#         agents=[retrieval_agent, forecasting_agent], 
#         tasks=[task],
#         verbose=True
#     )
    
#     try:
#         result = crew.kickoff()
#         return result
#     except Exception as e:
#         return f"Error processing query: {str(e)}"

# # --- Chat loop ---
# if __name__ == "__main__":
#     print("🌤️  Weather Information System")
#     print("Ask about weather patterns, forecasts, or current weather conditions!")
#     print("Examples:")
#     print("- 'What is the current weather in Mumbai?'")
#     print("- 'Tell me about monsoon patterns in India'")
#     print("- 'Weather forecast for Delhi'")
#     print("Type 'exit' or 'quit' to stop.\n")
    
#     while True:
#         try:
#             query = input("🌦️  Ask about weather: ").strip()
#             if query.lower() in ['exit', 'quit', '']:
#                 print("Goodbye! 👋")
#                 break
                
#             print("\n🔍 Processing your query...")
#             answer = ask_question(query)
#             print(f"\n📋 Answer: {answer}\n")
#             print("-" * 60)
            
#         except KeyboardInterrupt:
#             print("\n\nGoodbye! 👋")
#             break
#         except Exception as e:
#             print(f"\n❌ An error occurred: {str(e)}")
#             print("Please try again.\n")











from crewai import Agent, Task, Crew, Process
from crewai_tools import PDFSearchTool
from dotenv import load_dotenv

load_dotenv()

pdf_search_tool = PDFSearchTool(
    pdf_url="https://internal.imd.gov.in/section/nhac/dynamic/allindianew.pdf",
    # embedding_model="all-MiniLM-L6-v2",
    # collection_name="weather_rag"
)

research_agent = Agent(
    role="Research agent",
    goal="Search through the PDF to find relevant answers",
    allow_delegation=False,
    verbose=True,
    backstory=(
        "The research agent is adept at searching and extracting data from documents, ensuring accurate and prompt responses."
    ),
    tools=[pdf_search_tool],
)

professional_writer_agent=Agent(
    role="Professional writer",
    goal="Write professional emails based on the research agent's findings",
    allow_delegation=False,
    verbose=True,
    backstory=(
        "The professional writer specializes in transforming raw data into polished, engaging content that is easy to understand."
    ),
    tools=[],
)

answer_customer_question_task = Task(
    description=(
    """
    Answer the customer's questions based on the given PDF.
    The research agent will search through the PDF to find the relevant answers
    Your final answer MUST be clear and accurate, based on the context provided by the research agent.
    
    Here is the customer's question:
    {customer_question}
    """
    ),
    expected_output=("""
    Your final answer MUST be clear and accurate, based on the context provided by the research agent.
    """),
    tools=[pdf_search_tool],  # Use the PDF search tool to find relevant information
    agent=research_agent,
)

write_email_task = Task(
    description=(
        """
        Write a professional email based on the research agent's findings.
        The email should be clear, concise, and tailored to the customer's question.
        
        Here is the research agent's response:
        {research_agent_response}
        """
    ),
    expected_output=("""
    A well-structured professional email that addresses the customer's question.
    """),
    tools=[],
    agent=professional_writer_agent,
)

crew = Crew(
    agents=[research_agent, professional_writer_agent],
    tasks=[answer_customer_question_task, write_email_task],
    process=Process.sequential,  # Ensure tasks are executed in order   
)

customer_question = input("Enter the customer's question: ")
results=crew.kickoff(inputs={"customer_question": customer_question})
print(results)