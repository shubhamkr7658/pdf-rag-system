# pdf-rag-system
# 📄 PDF RAG System

A Local Retrieval-Augmented Generation (RAG) application built with **Node.js**, **LangChain.js**, **Qdrant Vector Database**, **Google Gemini Embeddings**, and a **locally hosted Qwen 3 4B model running through LM Studio**.

The application allows users to upload a PDF, convert its content into vector embeddings, store them in Qdrant, retrieve the most relevant document chunks using semantic search, and generate accurate answers using a local Large Language Model (LLM).

---

## 🚀 Features

- 📄 Extracts text from PDF files
- ✂️ Splits documents into semantic chunks
- 🧠 Generates embeddings using Gemini Embeddings
- 🗄️ Stores embeddings in Qdrant Vector Database
- 🔍 Performs semantic similarity search
- 🤖 Uses a locally hosted Qwen 3 model via LM Studio
- 💬 Answers user questions based only on the uploaded PDF

---

## 🛠 Tech Stack

- Node.js
- Express.js
- LangChain.js
- Qdrant Vector Database
- Google Gemini Embeddings
- LM Studio
- Qwen 3 4B
- pdf-parse

---

## 🏗 Project Workflow

```
PDF
 │
 ▼
PDF Parsing
 │
 ▼
Text Splitting
 │
 ▼
Gemini Embeddings
 │
 ▼
Qdrant Vector Database
 │
 ▼
Similarity Search
 │
 ▼
Relevant Context
 │
 ▼
Qwen 3 (LM Studio)
 │
 ▼
Answer
```

---

## 📷 Demo

The application can answer questions directly from uploaded PDF documents using Retrieval-Augmented Generation (RAG).

Example:

**Question**

```
What is the price of shampoo 200ml?
```

**Answer**

```
Price of Shampoo 200 ml: ₹220

Description:
Herbal shampoo for daily use.
```

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/pdf-rag-system.git

cd pdf-rag-system

npm install
```

Create a `.env` file:

```env
GOOGLE_API_KEY=YOUR_GEMINI_API_KEY
QDRANT_URL=http://localhost:6333
```

Start LM Studio and load a Qwen model.

Run the application:

```bash
npm run dev
```

---

## 📚 What I Learned

- Retrieval-Augmented Generation (RAG)
- Vector Embeddings
- Semantic Search
- Vector Databases
- Prompt Engineering
- Integrating LangChain with Local LLMs
- Running LLMs locally using LM Studio

---

## 🔮 Future Improvements

- Multiple PDF support
- Chat history
- Conversation memory
- Web interface
- Source citations
- Docker deployment

---

## 👨‍💻 Author

**Shubham Kumar**

Feel free to connect with me on LinkedIn and GitHub.
