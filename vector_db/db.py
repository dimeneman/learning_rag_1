from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

load_dotenv()

# ----------------------------------------------------
# Load Document
# ----------------------------------------------------
loader = TextLoader("document_loaders/note.txt")
docs = loader.load()

# ----------------------------------------------------
# Split Document
# ----------------------------------------------------
splitter = CharacterTextSplitter(
    separator="",
    chunk_size=100,
    chunk_overlap=1,
)

chunks = splitter.split_documents(docs)

print(f"Total Chunks Created: {len(chunks)}")

# ----------------------------------------------------
# Embedding Model
# ----------------------------------------------------
embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

# ----------------------------------------------------
# Create Vector Store
# ----------------------------------------------------
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chromaDB",
)

# ----------------------------------------------------
# Similarity Search
# ----------------------------------------------------
query = "House price prediction is which type of ML algorithm?"

results = vector_store.similarity_search_with_score(
    query=query,
    k=2
)

# ----------------------------------------------------
# Display Results
# ----------------------------------------------------
console = Console()

console.print(Rule("[bold cyan]Similarity Search Results[/bold cyan]"))

for i, (doc, score) in enumerate(results, start=1):

    console.print(
        Panel.fit(
            f"[bold yellow]Score:[/bold yellow] {score:.4f}\n"
            f"[bold yellow]Source:[/bold yellow] {doc.metadata.get('source')}\n"
            f"[bold yellow]Characters:[/bold yellow] {len(doc.page_content)}",
            title=f"[bold green]Result {i}[/bold green]",
            border_style="green",
        )
    )

    console.print(
        Panel(
            doc.page_content,
            title="[bold blue]Retrieved Chunk[/bold blue]",
            border_style="blue",
        )
    )

    console.print()