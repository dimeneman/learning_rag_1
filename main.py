from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.markdown import Markdown

load_dotenv()

console = Console()

# ==========================================================
# Load Embedding Model
# ==========================================================

console.print("[bold cyan]Loading Embedding Model...[/bold cyan]")

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

console.print("[green]✓ Embedding Model Loaded[/green]\n")

# ==========================================================
# Load Existing Chroma Database
# ==========================================================

console.print("[bold cyan]Loading Vector Database...[/bold cyan]")

vector_store = Chroma(
    persist_directory="chromaDB",
    embedding_function=embedding_model,
)

console.print("[green]✓ Vector Database Loaded[/green]\n")

# ==========================================================
# Retriever
# ==========================================================

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.7,
    },
)

# ==========================================================
# Gemini
# ==========================================================

console.print("[bold cyan]Loading Gemini...[/bold cyan]")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)

console.print("[green]✓ Gemini Loaded[/green]\n")

# ==========================================================
# Prompt
# ==========================================================

prompt = ChatPromptTemplate.from_template(
"""
You are an expert AI assistant.

Answer the user's question ONLY using the provided context.

If the answer cannot be found in the context, reply exactly:

I don't have enough information in the provided document.

Context:
{context}

Question:
{question}

Answer:
"""
)

# ==========================================================
# Welcome Message
# ==========================================================

console.print(
    Panel.fit(
        "[bold green]RAG System Ready![/bold green]\n\nType [cyan]0[/cyan] to exit.",
        title="System",
    )
)

# ==========================================================
# Chat Loop
# ==========================================================

while True:

    console.print()

    query = console.input("[bold yellow]You:[/bold yellow] ")

    if query == "0":
        break

    # ------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------

    docs = retriever.invoke(query)

    context = "\n\n".join(doc.page_content for doc in docs)

    console.print(Rule("[bold blue]Retrieved Chunks[/bold blue]"))

    for i, doc in enumerate(docs, start=1):

        console.print(
            Panel(
                doc.page_content,
                title=f"Chunk {i}",
                border_style="green",
            )
        )

    # ------------------------------------------------------
    # Prompt
    # ------------------------------------------------------

    final_prompt = prompt.invoke(
        {
            "context": context,
            "question": query,
        }
    )

    # ------------------------------------------------------
    # LLM
    # ------------------------------------------------------

    response = llm.invoke(final_prompt)

    # Extract only text
    if isinstance(response.content, str):
        answer = response.content
    else:
        answer = "\n".join(
            block["text"]
            for block in response.content
            if isinstance(block, dict) and "text" in block
        )

    console.print(Rule("[bold magenta]Answer[/bold magenta]"))

    console.print(
        Panel(
            Markdown(answer),
            title="Gemini",
            border_style="cyan",
        )
    )