
# 🏥 Medicure - AI-Powered Medical Chatbot  

**Medicure** is an AI-powered medical chatbot that extracts information from medical books in **PDF format**, stores the data in **Pinecone**, and uses **Gemini LLM** to provide intelligent responses. The chatbot retains previous conversations even after a page refresh, thanks to **PostgreSQL** for chat history storage.

---

## 🚀 Features  

- 📄 **PDF Processing**: Extracts data in chunks from medical books.  
- 🔍 **Vector Search**: Stores extracted data in **Pinecone** for efficient retrieval.  
- 🤖 **AI-Powered Chatbot**: Uses **Gemini LLM** to generate intelligent responses.  
- 💾 **Persistent Chat History**: Stores chats in **PostgreSQL** to prevent loss on refresh.  
- 🌐 **Web-Based Interface**: A seamless user experience for interacting with medical knowledge.  

---

## 🛠 Tech Stack  

- **Backend**: Python (FastAPI / Flask / Django)  
- **Frontend**: Streamlit
- **Database**: PostgreSQL  
- **Vector Database**: Pinecone  
- **LLM**: Gemini AI  

---

## ⚙️ Installation & Setup  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/ChandekarDhruvin/MediCure
cd Medicure
```

### 2️⃣ Install Dependencies  
```bash
pip install -r requirements.txt  # For backend
npm install  # If using a frontend framework
```

### 3️⃣ Set Up Environment Variables  
Create a `.env` file and add:  
```plaintext
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_env
POSTGRESQL_URL=your_postgres_db_url
GEMINI_API_KEY=your_gemini_api_key
```

### 4️⃣ Run the Application  
```bash
# Backend
python app.py  # or uvicorn main:app --reload if using FastAPI

# Frontend
npm run dev  # If using React / Next.js
```

---

## 🔥 Usage  

- Upload a medical **PDF file**  
- Chat with the AI chatbot  
- Get **accurate responses** based on extracted data  
- Refresh the page and continue the chat without losing history  

---

## 🏗️ Future Enhancements  

- ✅ Support for multiple PDF uploads  
- ✅ Fine-tuning the chatbot responses  
- ✅ Adding more LLM models for better accuracy  

---

## 📝 Contributing  

Feel free to **fork** this repository and submit pull requests. Contributions are welcome!  

---

## 📞 Contact  

For any queries, reach out to me at:  
📧 **Email**: dhruvinchandekar@gmail.com  
🔗 **GitHub**: https://github.com/ChandekarDhruvin

---

Would you like any modifications or additional sections? 😊

![image](https://github.com/user-attachments/assets/6d3559b7-41a6-4e3b-95b7-17ed484c8632)
