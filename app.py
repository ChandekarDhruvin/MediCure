# from flask import Flask,render_template,jsonify,request
# from src.helper import download_hugging_face_embeddings
# from langchain_pinecone import PineconeVectorStore
# from langchain_openai import OpenAI
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatMessagePromptTemplate
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.prompts import ChatPromptTemplate
# from dotenv import load_dotenv
# from src.prompt import *
# import os



# app = Flask(__name__)

# PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
# GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
# os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

# embeddings = download_hugging_face_embeddings()
# index_name = "curenow1"

# docresearch = PineconeVectorStore.from_existing_index(
#     index_name=index_name,
#     embedding=embeddings
# )


# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.0-flash-lite-preview-02-05",
#     temperature=0,
#     google_api_key = GEMINI_API_KEY,
#     max_tokens=None,
#     timeout=None,
#     max_retries=2,
#     # other params...
# )


# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system",system_prompt),
#         ("human","{input}"),
#     ]
# )

# retriever = docresearch.as_retriever(search_type = 'similarity',search_kwargs = {"k" : 3})

# question_answer_chain = create_stuff_documents_chain(llm,prompt)
# rag_chain = create_retrieval_chain(retriever,question_answer_chain)
# # rag_chain = create_retrieval_chain(pinecone_vector_store,question_answer_chain)


# @app.route("/")
# def index():
#     return render_template('chat.html')


# @app.route("/get",methods = ["GET","POST"])
# def chat():
#     msg = request.form["msg"]
#     input = msg
#     print(input)
#     response = rag_chain.invoke({"input" : msg})
#     print("Response : ",response["answer"])
#     return str(response["answer"])

# if __name__ == '__main__':
#     app.run(host="0.0.0.0",port = 8080,debug=True)

from flask import Flask, render_template, request, session, g, make_response, jsonify
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os
import psycopg2
from psycopg2.extras import DictCursor
import secrets
import logging

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", secrets.token_hex(32))

# Configure logging
logging.basicConfig(level=logging.INFO)

# Database Connection
DATABASE_URL = os.getenv("DATABASE_URL")

def get_db():
    """Connect to the PostgreSQL database."""
    if 'db' not in g:
        try:
            g.db = psycopg2.connect(DATABASE_URL)
        except Exception as e:
            logging.error(f"Database connection error: {e}")
            g.db = None
    return g.db

@app.teardown_appcontext
def close_db(error=None):
    """Close the database connection."""
    db = g.pop('db', None)
    if db is not None:
        db.close()

# Initialize Pinecone and LLM
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

embeddings = download_hugging_face_embeddings()
docresearch = PineconeVectorStore.from_existing_index(
    index_name="curenow1",
    embedding=embeddings
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite-preview-02-05",
    temperature=0,
    google_api_key=GEMINI_API_KEY,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

retriever = docresearch.as_retriever(search_type='similarity', search_kwargs={"k": 3})
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# --- Database Functions ---
def create_messages_table():
    """Create messages table if it does not exist."""
    db = get_db()
    if db:
        try:
            with db.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS messages (
                        id SERIAL PRIMARY KEY,
                        session_id TEXT,
                        user_message TEXT,
                        bot_response TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                db.commit()
                logging.info("Messages table verified or created successfully.")
        except Exception as e:
            logging.error(f"Error creating table: {e}")

def save_chat(session_id, user_message, bot_response):
    """Save user and bot messages to the database."""
    db = get_db()
    if db:
        try:
            with db.cursor() as cur:
                cur.execute("""
                    INSERT INTO messages (session_id, user_message, bot_response)
                    VALUES (%s, %s, %s)
                """, (session_id, user_message, bot_response))
                db.commit()
                logging.info(f"Chat saved: {user_message} -> {bot_response}")
        except Exception as e:
            logging.error(f"Error saving chat: {e}")

def get_chat_history(session_id):
    """Retrieve chat history for a given session."""
    db = get_db()
    if db:
        try:
            with db.cursor(cursor_factory=DictCursor) as cur:
                cur.execute("""
                    SELECT user_message, bot_response, timestamp 
                    FROM messages
                    WHERE session_id = %s 
                    ORDER BY timestamp ASC
                """, (session_id,))
                return cur.fetchall()
        except Exception as e:
            logging.error(f"Error retrieving chat history: {e}")
    return []

@app.route("/")
def index():
    """Render the chat interface and retrieve previous chat history."""
    session_id = request.cookies.get("session_id") or secrets.token_hex(16)
    
    # Get chat history using the function
    chat_history = get_chat_history(session_id)
    
    # Format the history for display, checking for missing timestamps
    formatted_history = []
    for msg in chat_history:
        timestamp = msg["timestamp"]
        formatted_history.append({
            "user": msg["user_message"],
            "bot": msg["bot_response"],
            "timestamp": timestamp.strftime("%I:%M %p") if timestamp else "Unknown"
        })

    response = make_response(render_template("chat.html", chat_history=formatted_history))
    response.set_cookie("session_id", session_id, max_age=60*60*24*7)  # 7 days
    
    return response

@app.route("/get", methods=["POST"])
def chat():
    """Handle chat input, get response, and save chat history."""
    session_id = request.cookies.get("session_id") or secrets.token_hex(16)
    msg = request.form.get("msg")
    
    if not msg:
        return jsonify({"error": "Invalid request"}), 400
    
    response = rag_chain.invoke({"input": msg})
    bot_response = response.get("answer", "Sorry, I couldn't process that request.")
    
    save_chat(session_id, msg, bot_response)
    
    return jsonify({"user": msg, "bot": bot_response})

if __name__ == '__main__':
    with app.app_context():
        create_messages_table()
    app.run(host="0.0.0.0", port=8080, debug=False)
