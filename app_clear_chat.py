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
from spellchecker import SpellChecker  

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", secrets.token_hex(32))

logging.basicConfig(level=logging.INFO)

DATABASE_URL = os.getenv("DATABASE_URL")

# Database Connection
def get_db():
    if 'db' not in g:
        try:
            g.db = psycopg2.connect(DATABASE_URL)
        except Exception as e:
            logging.error(f"Database connection error: {e}")
            g.db = None
    return g.db

@app.teardown_appcontext
def close_db(error=None):
    db = g.pop('db', None)
    if db is not None:
        db.close()

# Initialize Pinecone and LLM
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

embeddings = download_hugging_face_embeddings()
docresearch = PineconeVectorStore.from_existing_index("curenow1", embedding=embeddings)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite-preview-02-05",
    temperature=0,
    google_api_key=GEMINI_API_KEY,
    max_retries=2,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

retriever = docresearch.as_retriever(search_type='similarity', search_kwargs={"k": 3})
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Create Table
def create_messages_table():
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
        except Exception as e:
            logging.error(f"Error creating table: {e}")

# Save Chat
def save_chat(session_id, user_message, bot_response):
    db = get_db()
    if db:
        try:
            with db.cursor() as cur:
                cur.execute("""
                    INSERT INTO messages (session_id, user_message, bot_response)
                    VALUES (%s, %s, %s)
                """, (session_id, user_message, bot_response))
                db.commit()
        except Exception as e:
            logging.error(f"Error saving chat: {e}")

# Get Chat History
def get_chat_history(session_id):
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
    session_id = request.cookies.get("session_id", secrets.token_hex(16))
    chat_history = get_chat_history(session_id)
    formatted_history = [
        {"user": msg["user_message"], "bot": msg["bot_response"], "timestamp": msg["timestamp"].strftime("%I:%M %p") if msg["timestamp"] else "Unknown"}
        for msg in chat_history
    ]
    response = make_response(render_template("chat.html", chat_history=formatted_history))
    response.set_cookie("session_id", session_id, max_age=60*60*24*7)  
    return response

spell = SpellChecker()

@app.route("/get", methods=["POST"])
def chat():
    session_id = request.cookies.get("session_id", secrets.token_hex(16))
    msg = request.form.get("msg", "").strip()
    
    if not msg:
        return jsonify({"error": "Invalid request"}), 400
    
    msg_corrected = ' '.join([spell.correction(word) or word for word in msg.split()])
    
    if not msg_corrected.strip():
        return jsonify({"error": "Invalid request"}), 400
    
    try:
        response = rag_chain.invoke({"input": msg_corrected})
        bot_response = response.get("answer", "Sorry, I couldn't process that request.")
    except Exception as e:
        logging.error(f"Error processing chat: {e}")
        bot_response = "I'm experiencing issues. Please try again later."
    
    save_chat(session_id, msg, bot_response)
    return jsonify({"user": msg_corrected, "bot": bot_response})

@app.route("/clear_chats", methods=["POST"])
def clear_chats():
    db = get_db()
    if db:
        try:
            with db.cursor() as cur:
                cur.execute("TRUNCATE TABLE messages RESTART IDENTITY;")
                db.commit()
                logging.info("All chats cleared successfully.")
        except Exception as e:
            logging.error(f"Error clearing chat history: {e}")
            return jsonify({"error": "Failed to clear chat history"}), 500
    return jsonify({"success": True}), 200

if __name__ == '__main__':
    with app.app_context():
        create_messages_table()
    app.run(host="0.0.0.0", port=8080, debug=False)
