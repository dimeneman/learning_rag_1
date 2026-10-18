from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# ----------------------------------------------------
# Load PDF
# ----------------------------------------------------
loader = PyPDFLoader("document_loaders/mlops.pdf")
docs = loader.load()

# ----------------------------------------------------
# Split into Chunks
# ----------------------------------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

chunks = splitter.split_documents(docs)

print(f"Total Chunks Created: {len(chunks)}")

# ----------------------------------------------------
# Local Embedding Model
# ----------------------------------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

# ----------------------------------------------------
# Create Chroma Vector Database
# ----------------------------------------------------
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chromaDB",
)

print("\nVector Database created successfully!")