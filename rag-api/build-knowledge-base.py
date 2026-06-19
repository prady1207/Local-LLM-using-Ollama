import chromadb
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)

# Saving Data to disk so that it survives restart
client = chromadb.PersistentClient(path="./chromadb")

# Connecting to Ollama Embedding Function to convert text to vectors
em = OllamaEmbeddingFunction(
    model_name = "nomic-embed-text",
    url = "http://localhost:11434", # Ollama's default local address
)

# Create a new collection or reuse an existing collection (when not running for the first time)
collection = client.get_or_create_collection(
    name = "personal_profile",
    embedding_function = em, # Tells Chromadb how to convert text to vectors
)
# To delete an embedding from the collection

# collection.delete(ids="id_0")
# print(f"Number of vector in collection are {collection.count()}")

# Reading the text file with profile data
file_path = "profile.txt"
with open(file_path, "r", encoding="utf8") as f:
    raw_text = f.read()

# Splitting data into chunks based on newline
chunks = [chunk.strip() for chunk in raw_text.split("\n") if chunk.strip()]

# Generating distinct IDs and metadata mapping for each chunk
ids = [f"id_{i}" for i in range(len(chunks))]
metadatas = [{"source": file_path, "index": i} for i in range(len(chunks))]

# Adding raw text fragments to the collection
collection.add(
    ids=ids,
    documents=chunks,
    metadatas=metadatas
)

print(f"Successfully injected {len(chunks)} text chunks into Chroma DB!")