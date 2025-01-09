'''===========>>  FINAL WORKING FOR Q AND A ON TEXT PDF USING URL OF BOTH =============>'''


import os
import requests
import tempfile
from io import BytesIO

from flask import Flask, request, jsonify
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_unstructured import UnstructuredLoader

from langchain.document_loaders import PyPDFLoader, CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import logging
import openai
#from pyngrok import ngrok

# Set your OpenAI API key

import os
openai.api_key = os.getenv("OPENAI_API_KEY")

# Set up logging for error tracking
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Set Ngrok authtoken
#ngrok.set_auth_token("2qzSERs6P1tPui4MKH5mxcyU2sg_24ypr8YvfSMPWDaJdSRRc")

# Flask app setup
your_application = Flask(__name__)

# Define max token limit
MAX_TOKENS = 16385  # Maximum tokens for a model like GPT-3.5

# Function to download a file from a URL and save it to /kaggle/working directory
def download_file(url):
    try:
        # Check if the URL is from Dropbox and modify for direct download
        if "dropbox.com" in url:
            url = url.replace("?dl=0", "?dl=1")  # Change dl=0 to dl=1 for direct download
        response = requests.get(url)
        if response.status_code == 200:
            file_name = url.split("/")[-1]
            file_path = os.path.join("/tmp", file_name)
            with open(file_path, "wb") as f:
                f.write(response.content)
            return file_path
        else:
            logger.error(f"Failed to download file from {url} with status code {response.status_code}")
            raise Exception(f"Failed to download file from {url}")
    except Exception as e:
        logger.error(f"Error downloading file from {url}: {str(e)}")
        raise

# Function to load PDF file (from URL or local path)
def load_pdf(file_path):
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        return documents
    except Exception as e:
        logger.error(f"Error loading PDF file: {str(e)}")
        raise

# Function to load CSV file (from URL or local path)
def load_csv(file_path):
    try:
        loader = CSVLoader(file_path=file_path)
        documents = loader.load()
        return documents
    except Exception as e:
        logger.error(f"Error loading CSV file: {str(e)}")
        raise

# Function to load .txt file (from URL or local path) using UnstructuredLoader
def load_txt(file_path):
    try:
        loader = UnstructuredLoader([file_path])
        documents = loader.load()
        return documents
    except Exception as e:
        logger.error(f"Error loading TXT file: {str(e)}")
        raise

# Function to load website URL using UnstructuredURLLoader
def load_url(url):
    try:
        loader = UnstructuredURLLoader(urls=[url])
        documents = loader.load()
        return documents
    except Exception as e:
        logger.error(f"Error loading website: {str(e)}")
        raise

# Function to chunk documents recursively to avoid exceeding token limits
def chunk_documents(documents, max_tokens=MAX_TOKENS):
    chunked_documents = []
    current_chunk = []
    current_tokens = 0
    
    for doc in documents:
        doc_tokens = len(doc.page_content.split())
        
        if current_tokens + doc_tokens > max_tokens:
            if current_chunk:
                chunked_documents.append(current_chunk)
            current_chunk = [doc]
            current_tokens = doc_tokens
        else:
            current_chunk.append(doc)
            current_tokens += doc_tokens
    
    if current_chunk:
        chunked_documents.append(current_chunk)
    
    return chunked_documents

# Function to setup Q&A using LangChain
def setup_qa(documents):
    try:
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(documents, embeddings)
        retriever = vectorstore.as_retriever()
        qa_chain = ConversationalRetrievalChain.from_llm(
            ChatOpenAI(model="gpt-3.5-turbo"), retriever
        )
        return qa_chain
    except Exception as e:
        logger.error(f"Error setting up Q&A: {str(e)}")
        raise

@your_application.route("/chat", methods=["POST"])
def chat():
    try:
        query = request.json.get("question")
        file_type = request.json.get("file_type")
        file_url_or_path = request.json.get("file_url_or_path")

        if not query or not file_url_or_path or not file_type:
            return jsonify({"error": "Please provide a question, file URL or path, and file type"}), 400

        # Download file to /kaggle/working directory
        downloaded_file_path = download_file(file_url_or_path)

        if file_type == "pdf":
            documents = load_pdf(downloaded_file_path)
        elif file_type == "csv":
            documents = load_csv(downloaded_file_path)
        elif file_type == "txt":
            documents = load_txt(downloaded_file_path)
        elif file_type == "url":
            documents = load_url(downloaded_file_path)
        else:
            return jsonify({"error": "Unsupported file type"}), 400

        # Chunk the documents to avoid exceeding the token limit
        document_chunks = chunk_documents(documents)
        answers = []

        for chunk in document_chunks:
            qa_chain = setup_qa(chunk)
            response = qa_chain({"question": query, "chat_history": []})
            answers.append(response["answer"])

        return jsonify({"answer": " ".join(answers)})

    except Exception as e:
        logger.error(f"Error in chat route: {str(e)}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@your_application.route("/chatbot", methods=["POST"])
def chatbot():
    """
    Endpoint for normal chatting with an LLM.
    Accepts a JSON request with a 'question' parameter and responds with the LLM-generated answer.
    """
    try:
        query = request.json.get("question")
        if not query:
            return jsonify({"error": "Please provide a question"}), 400

        # Use the ChatOpenAI model to generate a response
        chat_model = ChatOpenAI(model="gpt-3.5-turbo")
        
        # Call the LLM with a simplified prompt structure
        response = chat_model.predict(query)

        return jsonify({"answer": response})
    except Exception as e:
        logger.error(f"Error in chatbot route: {str(e)}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500



if __name__ == "__main__":
    your_application.run(host="0.0.0.0", port=5000,debug=True)

   # public_url = ngrok.connect(5000)
    #print(f"Ngrok tunnel opened at: {public_url}")
    #app.run(debug=True, use_reloader=False)
