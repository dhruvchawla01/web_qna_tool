from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
from pathlib import Path
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
import numpy as np



# Load environment variables
load_dotenv(Path(__file__).parent / "key.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("⚠️ OpenAI API Key is missing. Set it in your environment or .env file.")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# In-memory store for web content
documents = []
vectorstore = None  # FAISS vector store will be initialized here

# Function to scrape text from a webpage
def scrape_website(url):
    try:
        print(f"🔍 Fetching content from: {url}")  # Debugging log

        response = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()  # Raise an error for HTTP issues

        soup = BeautifulSoup(response.text, "html.parser")

        # Extract text from <p> tags
        text = " ".join([p.get_text(strip=True) for p in soup.find_all("p")])

        if not text.strip():
            return f"Error: No readable text found at {url} (Possible JavaScript-based content)"

        print(f"✅ Successfully fetched {len(text)} characters from {url}")
        return text

    except requests.exceptions.Timeout:
        return f"Error: Timeout while accessing {url}"
    except requests.exceptions.RequestException as e:
        return f"Error: Failed to fetch {url} - {str(e)}"


# Route to ingest URLs and store content
@app.route('/ingest', methods=['POST'])
def ingest():
    global vectorstore, documents

    data = request.json
    urls = data.get("urls", [])

    if not urls:
        return jsonify({"error": "No URLs provided"}), 400

    # Scrape text from each URL
    documents = []
    for url in urls:
        text = scrape_website(url)
        if text.startswith("Error"):
            continue
        documents.append(text)

    if not documents:
        return jsonify({"error": "All URLs failed to ingest"}), 500

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_text("\n".join(documents))

    # Initialize OpenAI embeddings model
    embedding_model = OpenAIEmbeddings()

    # Generate embeddings
    embeddings = embedding_model.embed_documents(texts)

    # Convert texts to metadata format
    metadatas = [{"text": text} for text in texts]

    # Initialize FAISS vector store with embeddings and metadata
    vectorstore = FAISS.from_texts(texts, embedding_model, metadatas=metadatas)

    return jsonify({"message": f"Successfully ingested {len(urls)} URLs"}), 200

# Route to handle Q&A
@app.route('/ask', methods=['POST'])
def ask():
    global vectorstore
    data = request.get_json()

    print("Received request:", data)  # Debugging print

    if not data or "question" not in data:
        return jsonify({"error": "No question provided"}), 400

    if vectorstore is None:
        return jsonify({"error": "No content ingested yet"}), 400

    # Retrieve and generate answer using LangChain
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

    try:
        answer = qa_chain.run(data["question"])
    except Exception as e:
        return jsonify({"error": f"OpenAI API Error: {str(e)}"}), 500

    return jsonify({"answer": answer})

# Route for homepage
@app.route('/')
def home():
    return render_template("index.html")

# Dummy fetch endpoint to avoid 404 errors
@app.route('/fetch', methods=['POST'])
def fetch():
    return jsonify({"message": "Fetch endpoint is not implemented"}), 200

if __name__ == '__main__':
    app.run(debug=True)
