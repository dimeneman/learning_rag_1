from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    separator="",
    chunk_size = 10,
    chunk_overlap = 1,
)

data = TextLoader("document_loaders/note.txt")
docs = data.load()

chunks = splitter.split_documents(docs)

print(f"Total number of chunks: {len(chunks)}")
print()

for chunk in chunks:
    print("Chunk ----------->")
    print(chunk.page_content)
