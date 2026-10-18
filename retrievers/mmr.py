from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

# ----------------------------------------------------
# Console
# ----------------------------------------------------
console = Console()

# ----------------------------------------------------
# Documents
# ----------------------------------------------------
docs = [
    Document(page_content="Gradient descent is an optimization algorithm used in machine learning."),
    Document(page_content="Gradient descent minimizes the loss function."),
    Document(page_content="Gradient descent is an optimization algorithm that minimizes the loss function."),
    Document(page_content="Neural networks use gradient descent for training."),
    Document(page_content="Support Vector Machines are supervised learning algorithms."),
]

# ----------------------------------------------------
# Embedding Model
# ----------------------------------------------------
console.print("[bold cyan]Loading Embedding Model...[/bold cyan]")

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

console.print("[bold green]✓ Embedding Model Loaded[/bold green]\n")

# ----------------------------------------------------
# Create Vector Store
# ----------------------------------------------------
console.print("[bold cyan]Creating Chroma Vector Database...[/bold cyan]")

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
)

console.print("[bold green]✓ Vector Database Created[/bold green]\n")

# ----------------------------------------------------
# Similarity Retriever
# ----------------------------------------------------
similarity_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)

query = "What is gradient descent?"

console.print(Rule("[bold blue]Similarity Search[/bold blue]"))
console.print(f"[bold yellow]Query:[/bold yellow] {query}\n")

similarity_docs = similarity_retriever.invoke(query)

for i, doc in enumerate(similarity_docs, start=1):
    console.print(
        Panel(
            doc.page_content,
            title=f"[bold green]Result {i}[/bold green]",
            subtitle=f"Characters: {len(doc.page_content)}",
            border_style="green",
        )
    )

# ----------------------------------------------------
# MMR Retriever
# ----------------------------------------------------
mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3},
)

console.print()
console.print(Rule("[bold magenta]MMR Search[/bold magenta]"))
console.print(f"[bold yellow]Query:[/bold yellow] {query}\n")

mmr_docs = mmr_retriever.invoke(query)

for i, doc in enumerate(mmr_docs, start=1):
    console.print(
        Panel(
            doc.page_content,
            title=f"[bold cyan]Result {i}[/bold cyan]",
            subtitle=f"Characters: {len(doc.page_content)}",
            border_style="cyan",
        )
    )