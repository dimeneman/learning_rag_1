import os
from google import genai
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate

# For better output in terminal
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()
load_dotenv()


url = 'https://www.apple.com/in/macbook-pro/'

data = WebBaseLoader(url)
docs = data.load()


template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an expert AI assistant specialized in summarizing documents.

Your task is to generate a concise, accurate, and well-structured summary of the provided text.

Instructions:
- Read the entire document carefully.
- Capture the main ideas and important facts.
- Preserve the original meaning.
- Do not add information that is not present in the text.
- Avoid unnecessary repetition.
- Use clear and simple language.
- If the document contains multiple topics, summarize each briefly.
- If the input is very short, simply rewrite it more concisely.

Return only the summary.
            """,
        ),
        (
            "human",
            """
Document:

{document}
            """,
        ),
    ]
)

prompt = template.format_messages(document = docs[0].page_content)



model_name = "gemini-3.5-flash-lite"
llm = ChatGoogleGenerativeAI(
    model = model_name
)

response = llm.invoke(prompt)


console.print(
    Panel(
        Markdown(response.text),
        title="AI Summary",
        border_style="green"
    )
)
