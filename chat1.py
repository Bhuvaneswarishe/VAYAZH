import os
import requests
import certifi
import PyPDF2
from itertools import chain
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Milvus

# Milvus Config
MILVUS_URI = ".."
MILVUS_TOKEN = ".."
COLLECTION_NAME = "vayazh"

# ----------------------------- #
# Helpers
# ----------------------------- #

# Function to fetch content from a website
def fetch_website_content(url):
    response = requests.get(url, verify=certifi.where())
    return response.text

# Function to extract text from a PDF file
def extract_pdf_text(pdf_file):
    with open(pdf_file, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = "".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
    return text

# Split large text into manageable chunks
def split_text(text, chunk_size=500, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(text)

# ✅ Load the KisanVaani Hugging Face dataset
def load_kissanvani_data():
    dataset = load_dataset("KisanVaani/agriculture-qa-english-only")
    kissan_texts = []

    for item in dataset["train"]:
        question = item.get("question", "").strip()
        answer = item.get("answers", "").strip()
        if question and answer:
            kissan_texts.append(f"Q: {question}\nA: {answer}")

    return kissan_texts

# ----------------------------- #
# Vector Store Initialization
# ----------------------------- #
def initialize_vector_store(contents):
    try:
        # Use a local model path if available
        local_model_path = "./all-MiniLM-L6-v2"
        if os.path.exists(local_model_path):
            print("🔄 Loading model from local path...")
            model_path = local_model_path
        else:
            print("🌐 Loading model from HuggingFace...")
            model_path = "sentence-transformers/all-mpnet-base-v2"


        # Ensure SentenceTransformer can load without error before embedding
        _ = SentenceTransformer(model_path)

        embedding_function = HuggingFaceEmbeddings(model_name=model_path)

        # Load KisanVaani data
        kissan_texts = load_kissanvani_data()

        all_contents = contents + kissan_texts

        # Convert everything to chunks
        web_chunks = list(chain.from_iterable(split_text(content) for content in all_contents))

        print("✅ Connecting to Milvus...")
        db = Milvus.from_texts(
            texts=web_chunks,
            embedding=embedding_function,
            collection_name=COLLECTION_NAME,
            connection_args={
                "uri": MILVUS_URI,
                "token": MILVUS_TOKEN
            },
            text_field="text",        # 👈 use correct schema
            vector_field="embedding"  # 👈 use correct schema
        )

        print(f"✅ Data uploaded to Milvus collection: {COLLECTION_NAME}")
        return db

    except Exception as e:
        print("❌ Error initializing Milvus vector store:", str(e))
        raise e
