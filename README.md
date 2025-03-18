# Web Content Q&A Tool

A simple web-based tool that allows users to fetch web content and ask questions using OpenAI’s GPT models.

## Description
This tool extracts content from given URLs, processes it into a vector database, and answers user questions based on the extracted data using OpenAI’s GPT models. It uses **Flask**, **FAISS**, and **LangChain** for retrieval-augmented generation (RAG).

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/web-qna-tool.git
   cd web-qna-tool
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Mac/Linux
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   - Create a `.env` file in the root directory:
     ```
     OPENAI_API_KEY=your-api-key-here
     ```
   - **(Do NOT commit your `.env` file!)**

---

## How to Run

1. **Start the Flask server:**
   ```bash
   python main.py
   ```
2. Open your browser and go to:
   ```
   http://127.0.0.1:5000
   ```

---

## How to Use the API

### 1️. Ingest Content (Fetch URLs)
**Endpoint:**
```
POST /ingest
```
**Request Body:**
```json
{
  "urls": ["https://example.com"]
}
```
**Response:**
```json
{
  "message": "Content successfully ingested!"
}
```

### 2️. Ask a Question
**Endpoint:**
```
POST /ask
```
**Request Body:**
```json
{
  "question": "What is Model Context Protocol?"
}
```
**Response:**
```json
{
  "answer": "Model Context Protocol (MCP) is a method for integrating structured data into AI responses..."
}
```

---

## Features

Fetches web content from multiple URLs  
Stores extracted content in FAISS for retrieval  
Uses OpenAI’s GPT models for answering questions  
Supports **GPT-4o Mini** (configurable)  
Simple UI for easy interaction  

---

## Security Measures

**API Key Security:** The OpenAI API key is stored in a `.env` file and **not** hardcoded in the code.  
**Restricted Submission:** The `.env` file is **excluded** from submission using `.gitignore`.  
**Input Validation:** The application validates user input to prevent errors and abuse.  
**Error Handling:** Proper error messages are returned for invalid requests.  

---

## License
This project is licensed under the MIT License.

---

## Author
Developed by **Dhruv Chawla**  
Feel free to contribute or report issues!
