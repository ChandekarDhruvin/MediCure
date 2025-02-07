import streamlit as st
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
import os

from flask import Flask,render_template,jsonify,request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatMessagePromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

# Load environment variables
load_dotenv()


app = Flask(__name__)

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

embeddings = download_hugging_face_embeddings()
index_name = "curenow1"

docresearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)


llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite-preview-02-05",
    temperature=0,
    google_api_key = GEMINI_API_KEY,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)


prompt = ChatPromptTemplate.from_messages(
    [
        ("system",system_prompt),
        ("human","{input}"),
    ]
)

retriever = docresearch.as_retriever(search_type = 'similarity',search_kwargs = {"k" : 3})

question_answer_chain = create_stuff_documents_chain(llm,prompt)
rag_chain = create_retrieval_chain(retriever,question_answer_chain)
# rag_chain = create_retrieval_chain(pinecone_vector_store,question_answer_chain)


# Streamlit final code

# # ---- Streamlit UI ----
# st.set_page_config(page_title="Chat with AI", layout="wide")
# st.title("💬 CureNow AI Chatbot")
# st.write("Ask me anything!")

# # Initialize chat history
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Display previous messages
# for msg in st.session_state.messages:
#     with st.chat_message(msg["role"]):
#         st.write(msg["content"])

# # User input
# user_input = st.chat_input("Enter your question:")

# if user_input:
#     with st.chat_message("user"):
#         st.write(user_input)

#     # Process user input
#     with st.spinner("Thinking..."):
#         response = rag_chain.invoke({"input": user_input})
#         bot_response = response["answer"]

#     # Display AI response
#     with st.chat_message("assistant"):
#         st.write(bot_response)

#     # Store message history
#     st.session_state.messages.append({"role": "user", "content": user_input})
#     st.session_state.messages.append({"role": "assistant", "content": bot_response})

import streamlit as st
import base64
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
import os

# Set page config as the first Streamlit command
st.set_page_config(page_title="Chat with AI", layout="wide")

# Load environment variables
load_dotenv()

st.title("💬 CureNow AI Chatbot")
st.write("Ask me anything!")

# Add background image with reduced opacity for readability
background_image_url = "static\\bot2.jpg"  # Replace with actual path or URL to your image
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{base64.b64encode(open('static/bot2.jpg', 'rb').read()).decode()}");
        background-size: cover;  /* Ensures image scales properly with screen size */
        background-position: center center; /* Centers the background */
        height: 100vh;  /* Ensures the background image covers the full viewport height */
        background-repeat: no-repeat;
        position: relative;
    }}
    .stApp::before {{
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0, 0, 0, 0.4); /* Dark overlay with low opacity */
        z-index: 1;
    }}
    .stTitle, .stWrite {{
        color: #ffffff;  /* White color for title and prompt text */
        background-color: rgba(0, 0, 0, 0.5); /* Slight dark background for readability */
        padding: 10px;
        border-radius: 8px;
        z-index: 2;
    }}
    .stChatMessage {{
        color: black;  /* Black color for chat messages */
        z-index: 2;
    }}
    .stTextInput input {{
        background-color: rgba(255, 255, 255, 0.8); /* Transparent input box */
        color: black;
        z-index: 2;
    }}
    .stTextInput label {{
        color: black;
        z-index: 2;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User input
user_input = st.chat_input("Enter your question:")

if user_input:
    with st.chat_message("user"):
        st.write(user_input)

    # Process user input
    with st.spinner("Thinking..."):
        response = rag_chain.invoke({"input": user_input})
        bot_response = response["answer"]

    # Display AI response in bold
    with st.chat_message("assistant"):
        st.markdown(f"**{bot_response}**")  # This will make the response bold

    # Store message history
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": bot_response})