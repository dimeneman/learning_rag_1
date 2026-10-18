from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_google_genai import ChatGoogleGenerativeAI

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

load_dotenv()

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
# Create Chroma Vector Store
# ----------------------------------------------------
console.print("[bold cyan]Creating Chroma Vector Database...[/bold cyan]")

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
)

console.print("[bold green]✓ Vector Database Created[/bold green]\n")

# ----------------------------------------------------
# Base Retriever
# ----------------------------------------------------
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# ----------------------------------------------------
# Gemini LLM
# ----------------------------------------------------
console.print("[bold cyan]Loading Gemini...[/bold cyan]")

model_name = "gemini-3.5-flash-lite"
llm = ChatGoogleGenerativeAI(
    model=model_name
)

console.print("[bold green]✓ Gemini Loaded[/bold green]\n")

# ----------------------------------------------------
# Multi Query Retriever
# ----------------------------------------------------
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=llm,
)

query = "What is gradient descent?"

console.print(Rule("[bold magenta]Multi Query Retrieval[/bold magenta]"))
console.print(f"[bold yellow]Original Query:[/bold yellow] {query}\n")

results = multi_query_retriever.invoke(query)

for i, doc in enumerate(results, start=1):
    console.print(
        Panel(
            doc.page_content,
            title=f"[bold green]Retrieved Document {i}[/bold green]",
            subtitle=f"Characters: {len(doc.page_content)}",
            border_style="green",
        )
    )